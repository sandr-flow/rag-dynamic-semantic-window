"""Retrieval strategies for benchmark comparison."""

from abc import ABC, abstractmethod
from typing import Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document, NodeWithScore, QueryBundle, TextNode

from src.config import (
    DEFAULT_DYNAMIC_SEMANTIC_CONFIG,
    DEFAULT_EXPANSION_CONFIG,
    DEFAULT_FIXED_WINDOW_CONFIG,
    DEFAULT_NAIVE_CHUNKING_CONFIG,
    DEFAULT_RETRIEVAL_CONFIG,
    DEFAULT_SEMANTIC_SPLITTER_CONFIG,
)
from src.dynamic_retriever import DynamicSemanticExpander
from src.utils import create_sentence_nodes, split_into_sentences


class BaseStrategy(ABC):
    """Base class for retrieval strategies."""

    def __init__(self, documents: list[Document], top_k: int = 5):
        """
        Initialize strategy with documents.

        Args:
            documents: List of documents to index.
            top_k: Number of results to retrieve.
        """
        self.documents = documents
        self.top_k = top_k
        self.index: Optional[VectorStoreIndex] = None
        self._build_index()

    @abstractmethod
    def _build_index(self) -> None:
        """Build the vector index for this strategy."""
        pass

    @abstractmethod
    def retrieve(self, query: str) -> list[NodeWithScore]:
        """
        Retrieve relevant nodes for a query.

        Args:
            query: Query string.

        Returns:
            List of retrieved nodes with scores.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass


class NaiveChunkingStrategy(BaseStrategy):
    """
    Baseline strategy using fixed-size chunking.

    Uses SentenceSplitter with configurable chunk_size and overlap.
    """

    @property
    def name(self) -> str:
        return "Naive Chunking"

    def _build_index(self) -> None:
        """Build index with naive chunking."""
        splitter = SentenceSplitter(
            chunk_size=DEFAULT_NAIVE_CHUNKING_CONFIG.chunk_size,
            chunk_overlap=DEFAULT_NAIVE_CHUNKING_CONFIG.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve using simple top-k."""
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        return retriever.retrieve(query)


class FixedWindowStrategy(BaseStrategy):
    """
    Control strategy using fixed sentence window.

    Uses SentenceWindowNodeParser with configurable window_size.
    """

    @property
    def name(self) -> str:
        return "Fixed Window"

    def _build_index(self) -> None:
        """Build index with sentence window parser."""
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=DEFAULT_FIXED_WINDOW_CONFIG.window_size,
            window_metadata_key=DEFAULT_FIXED_WINDOW_CONFIG.window_metadata_key,
            original_text_metadata_key=DEFAULT_FIXED_WINDOW_CONFIG.original_text_metadata_key,
        )
        nodes = node_parser.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve with metadata replacement for window context."""
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query)

        # Replace node text with window context
        postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
        return postprocessor.postprocess_nodes(nodes)


class DynamicSemanticStrategy(BaseStrategy):
    """
    Experimental strategy using dynamic semantic window.

    Indexes each sentence separately and expands context based on
    cosine similarity of neighbors.
    
    Uses HYBRID WINDOW approach:
    - Always includes ±min_window neighbors (safety net)
    - Expands further only if similarity > threshold
    """

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        threshold: float = DEFAULT_EXPANSION_CONFIG.threshold,
        max_expand: int = DEFAULT_EXPANSION_CONFIG.max_expand,
        min_window: int = DEFAULT_EXPANSION_CONFIG.min_window,
        relevance_threshold_pct: float = DEFAULT_EXPANSION_CONFIG.relevance_threshold_pct,
        merge_gap: int = DEFAULT_EXPANSION_CONFIG.merge_gap,
        seed_rejection_log_path: str | None = None,
        phantom_window: int = DEFAULT_DYNAMIC_SEMANTIC_CONFIG.phantom_window,
        prefetch_multiplier: int = DEFAULT_DYNAMIC_SEMANTIC_CONFIG.prefetch_multiplier,
    ):
        """
        Initialize dynamic semantic strategy.

        Args:
            documents: List of documents to index.
            top_k: Number of results to retrieve.
            threshold: Cosine similarity threshold for expansion beyond min_window.
            max_expand: Maximum sentences to expand in each direction.
            min_window: Minimum neighbors to always include (hybrid safety net).
            relevance_threshold_pct: Query relevance threshold as % of max score (0.85 = 85%).
            merge_gap: Maximum gap between clusters to merge (2 = merge if <= 2 sentences apart).
            seed_rejection_log_path: Path to log rejected seeds (JSONL format).
            phantom_window: Number of neighbors to include in embedding context (0 = disabled).
            prefetch_multiplier: Multiplier for first-pass retrieval (4 = fetch top_k*4 seeds).
        """
        self.threshold = threshold
        self.max_expand = max_expand
        self.min_window = min_window
        self.relevance_threshold_pct = relevance_threshold_pct
        self.merge_gap = merge_gap
        self.seed_rejection_log_path = seed_rejection_log_path
        self.phantom_window = phantom_window
        self.prefetch_multiplier = prefetch_multiplier
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Dynamic Semantic"

    def _build_index(self) -> None:
        """Build index with per-sentence nodes using Phantom Embeddings."""
        # Combine all document text
        full_text = " ".join(doc.text for doc in self.documents)

        # Split into sentences and create linked nodes
        sentences = split_into_sentences(full_text)
        nodes = create_sentence_nodes(sentences)

        # Compute embeddings with phantom context
        embed_model = Settings.embed_model
        
        for i, node in enumerate(nodes):
            if self.phantom_window > 0:
                # Build phantom context: [prev...prev, CENTER, next...next]
                start_idx = max(0, i - self.phantom_window)
                end_idx = min(len(nodes), i + self.phantom_window + 1)
                
                context_texts = [nodes[j].text for j in range(start_idx, end_idx)]
                phantom_text = " ".join(context_texts)
                
                # Embedding from context, but node.text stays as single sentence
                node.embedding = embed_model.get_text_embedding(phantom_text)
            else:
                # Original behavior: embed single sentence
                node.embedding = embed_model.get_text_embedding(node.text)

        # Build index with storage context for docstore access
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        self.index = VectorStoreIndex(
            nodes, storage_context=storage_context, store_nodes_override=True
        )
        self.docstore = storage_context.docstore

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """
        Retrieve with dynamic semantic expansion using two-pass approach.
        
        First pass: Fetch top_k * prefetch_multiplier seeds for broad coverage.
        Second pass: Expand and deduplicate to target top_k results.
        """
        # Two-pass: first pass fetches more seeds for better coverage
        prefetch_k = self.top_k * self.prefetch_multiplier
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=prefetch_k)
        nodes = retriever.retrieve(query)

        # Apply dynamic expansion with hybrid window + Query-Aware + Multi-Seed Merging
        expander = DynamicSemanticExpander(
            docstore=self.docstore,
            threshold=self.threshold,
            max_expand=self.max_expand,
            min_window=self.min_window,
            relevance_threshold_pct=self.relevance_threshold_pct,
            merge_gap=self.merge_gap,
            seed_rejection_log_path=self.seed_rejection_log_path,
            target_clusters=self.top_k,  # Pass target for deduplication
        )
        return expander.postprocess_nodes(nodes, QueryBundle(query_str=query))


class SemanticSplitterStrategy(BaseStrategy):
    """
    Strategy using SemanticSplitterNodeParser for embeddings-based splitting.
    """

    @property
    def name(self) -> str:
        return "Semantic Splitter"

    def _build_index(self) -> None:
        """Build index with semantic splitter."""
        from llama_index.core.node_parser import SemanticSplitterNodeParser

        parser = SemanticSplitterNodeParser(
            buffer_size=DEFAULT_SEMANTIC_SPLITTER_CONFIG.buffer_size,
            breakpoint_percentile_threshold=DEFAULT_SEMANTIC_SPLITTER_CONFIG.breakpoint_percentile_threshold,
            embed_model=Settings.embed_model,
        )
        nodes = parser.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve using simple top-k."""
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        return retriever.retrieve(query)



