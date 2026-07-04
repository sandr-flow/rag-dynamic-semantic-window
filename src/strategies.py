"""Retrieval strategies for benchmark comparison."""

import re
from abc import ABC, abstractmethod

import numpy as np
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document, NodeWithScore, QueryBundle

from src.config import (
    DEFAULT_DYNAMIC_SEMANTIC_CONFIG,
    DEFAULT_EXPANSION_CONFIG,
    DEFAULT_FIXED_WINDOW_CONFIG,
    DEFAULT_NAIVE_CHUNKING_CONFIG,
    DEFAULT_RETRIEVAL_CONFIG,
    DEFAULT_SEMANTIC_SPLITTER_CONFIG,
    DEFAULT_TOKEN_TEXT_SPLITTER_CONFIG,
    DynamicSemanticConfig,
    ExpansionConfig,
    FixedWindowConfig,
    NaiveChunkingConfig,
    SemanticSplitterConfig,
    TokenTextSplitterConfig,
)
from src.dynamic_retriever import DynamicSemanticExpander
from src.utils import build_embedding_texts, create_sentence_nodes, split_into_sentences


def _safe_doc_id(raw_doc_id: object, fallback: str, used: set[str]) -> str:
    """Return a stable node-id-safe document id."""
    value = str(raw_doc_id or fallback)
    doc_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_") or fallback
    if doc_id not in used:
        used.add(doc_id)
        return doc_id

    suffix = 2
    while f"{doc_id}_{suffix}" in used:
        suffix += 1
    unique = f"{doc_id}_{suffix}"
    used.add(unique)
    return unique


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
        self.index: VectorStoreIndex | None = None
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

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        config: NaiveChunkingConfig | None = None,
    ):
        self.config = config or DEFAULT_NAIVE_CHUNKING_CONFIG
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Naive Chunking"

    def _build_index(self) -> None:
        """Build index with naive chunking."""
        splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
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

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        config: FixedWindowConfig | None = None,
    ):
        self.config = config or DEFAULT_FIXED_WINDOW_CONFIG
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Fixed Window"

    def _build_index(self) -> None:
        """Build index with sentence window parser."""
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.config.window_size,
            window_metadata_key=self.config.window_metadata_key,
            original_text_metadata_key=self.config.original_text_metadata_key,
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


class TokenTextSplitterStrategy(BaseStrategy):
    """
    Baseline strategy using LlamaIndex TokenTextSplitter.

    This is useful as an explicit token-based baseline next to SentenceSplitter.
    """

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        config: TokenTextSplitterConfig | None = None,
    ):
        self.config = config or DEFAULT_TOKEN_TEXT_SPLITTER_CONFIG
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Token Text Splitter"

    def _build_index(self) -> None:
        splitter = TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        return retriever.retrieve(query)


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
        phantom_window: int = DEFAULT_DYNAMIC_SEMANTIC_CONFIG.phantom_window,
        prefetch_multiplier: int = DEFAULT_DYNAMIC_SEMANTIC_CONFIG.prefetch_multiplier,
        dynamic_config: DynamicSemanticConfig | None = None,
        expansion_config: ExpansionConfig | None = None,
    ):
        """
        Initialize dynamic semantic strategy.

        All expansion parameters (threshold, skip_threshold, max_expand, etc.)
        are taken from DEFAULT_EXPANSION_CONFIG in config.py.

        Args:
            documents: List of documents to index.
            top_k: Number of results to retrieve.
            phantom_window: Number of neighbors to include in embedding context (0 = disabled).
            prefetch_multiplier: Multiplier for first-pass retrieval (4 = fetch top_k*4 seeds).
        """
        self.dynamic_config = dynamic_config or DynamicSemanticConfig(
            phantom_window=phantom_window,
            prefetch_multiplier=prefetch_multiplier,
        )
        self.expansion_config = expansion_config or DEFAULT_EXPANSION_CONFIG
        self.phantom_window = self.dynamic_config.phantom_window
        self.prefetch_multiplier = self.dynamic_config.prefetch_multiplier
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Dynamic Semantic"

    def _build_index(self) -> None:
        """Build index with per-sentence nodes using Phantom Embeddings."""
        nodes = []
        doc_ids = []
        all_embedding_texts: list[str] = []
        all_sentences: list[str] = []
        used_doc_ids: set[str] = set()
        embed_model = Settings.embed_model

        for doc_idx, doc in enumerate(self.documents):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            raw_doc_id = (
                metadata.get("source_doc")
                or metadata.get("id")
                or getattr(doc, "id_", None)
                or f"doc_{doc_idx}"
            )
            doc_id = _safe_doc_id(raw_doc_id, f"doc_{doc_idx}", used_doc_ids)
            doc_sentences = split_into_sentences(doc.text)
            doc_nodes = create_sentence_nodes(doc_sentences, doc_id=doc_id)

            for node in doc_nodes:
                node.metadata["source_doc"] = doc_id
                if "title" in metadata:
                    node.metadata["title"] = metadata["title"]

            nodes.extend(doc_nodes)
            doc_ids.extend([doc_id] * len(doc_sentences))
            all_sentences.extend(doc_sentences)
            # Phantom contexts never cross document boundaries
            all_embedding_texts.extend(
                build_embedding_texts(doc_sentences, self.phantom_window)
            )

        # One batched call instead of one model call per sentence
        embeddings = embed_model.get_text_embedding_batch(all_embedding_texts)
        for node, embedding in zip(nodes, embeddings, strict=True):
            node.embedding = embedding

        # Dual-space: adjacency sims always come from clean sentence
        # embeddings, while the index and query path stay on phantom
        # embeddings (phantom neighbors textually overlap, inflating
        # adjacency cosines). With phantom_window=0 the spaces coincide,
        # so no second batch is needed.
        adjacency_matrix = None
        if self.phantom_window > 0:
            adjacency_matrix = np.array(
                embed_model.get_text_embedding_batch(all_sentences), dtype=np.float32
            )

        # Hand the sentence arrays to the expander before index construction:
        # VectorStoreIndex strips embeddings from the nodes it stores, so the
        # docstore cannot serve as an embedding source afterwards.
        embeddings_matrix = np.array([node.embedding for node in nodes], dtype=np.float32)
        self.expander = DynamicSemanticExpander(
            sentences=[node.text for node in nodes],
            node_ids=[node.node_id for node in nodes],
            embeddings=embeddings_matrix,
            adjacency_embeddings=adjacency_matrix,
            doc_ids=doc_ids,
            target_clusters=self.top_k,
            threshold=self.expansion_config.threshold,
            skip_threshold=self.expansion_config.skip_threshold,
            max_expand=self.expansion_config.max_expand,
            min_window=self.expansion_config.min_window,
            min_chunk_length=self.expansion_config.min_chunk_length,
            relevance_threshold_pct=self.expansion_config.relevance_threshold_pct,
            merge_gap=self.expansion_config.merge_gap,
        )

        self.index = VectorStoreIndex(nodes)

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

        return self.expander.postprocess_nodes(nodes, QueryBundle(query_str=query))


class SemanticSplitterStrategy(BaseStrategy):
    """
    Strategy using SemanticSplitterNodeParser for embeddings-based splitting.
    """

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        config: SemanticSplitterConfig | None = None,
    ):
        self.config = config or DEFAULT_SEMANTIC_SPLITTER_CONFIG
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Semantic Splitter"

    def _build_index(self) -> None:
        """Build index with semantic splitter."""
        from llama_index.core.node_parser import SemanticSplitterNodeParser

        parser = SemanticSplitterNodeParser(
            buffer_size=self.config.buffer_size,
            breakpoint_percentile_threshold=self.config.breakpoint_percentile_threshold,
            embed_model=Settings.embed_model,
        )
        nodes = parser.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve using simple top-k."""
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        return retriever.retrieve(query)



