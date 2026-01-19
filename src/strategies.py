"""Retrieval strategies for benchmark comparison."""

from abc import ABC, abstractmethod
from typing import Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document, NodeWithScore, QueryBundle, TextNode

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

    Uses SentenceSplitter with chunk_size=256 and overlap=20.
    """

    @property
    def name(self) -> str:
        return "Naive Chunking"

    def _build_index(self) -> None:
        """Build index with naive chunking."""
        splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
        nodes = splitter.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve using simple top-k."""
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        return retriever.retrieve(query)


class FixedWindowStrategy(BaseStrategy):
    """
    Control strategy using fixed sentence window.

    Uses SentenceWindowNodeParser with window_size=3.
    """

    @property
    def name(self) -> str:
        return "Fixed Window"

    def _build_index(self) -> None:
        """Build index with sentence window parser."""
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
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
        top_k: int = 5,
        threshold: float = 0.6,
        max_expand: int = 5,
        min_window: int = 1,
    ):
        """
        Initialize dynamic semantic strategy.

        Args:
            documents: List of documents to index.
            top_k: Number of results to retrieve.
            threshold: Cosine similarity threshold for expansion beyond min_window.
            max_expand: Maximum sentences to expand in each direction.
            min_window: Minimum neighbors to always include (hybrid safety net).
        """
        self.threshold = threshold
        self.max_expand = max_expand
        self.min_window = min_window
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Dynamic Semantic"

    def _build_index(self) -> None:
        """Build index with per-sentence nodes."""
        # Combine all document text
        full_text = " ".join(doc.text for doc in self.documents)

        # Split into sentences and create linked nodes
        sentences = split_into_sentences(full_text)
        nodes = create_sentence_nodes(sentences)

        # Compute embeddings for all nodes
        embed_model = Settings.embed_model
        for node in nodes:
            node.embedding = embed_model.get_text_embedding(node.text)

        # Build index with storage context for docstore access
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        self.index = VectorStoreIndex(
            nodes, storage_context=storage_context, store_nodes_override=True
        )
        self.docstore = storage_context.docstore

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve with dynamic semantic expansion."""
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query)

        # Apply dynamic expansion with hybrid window
        expander = DynamicSemanticExpander(
            docstore=self.docstore,
            threshold=self.threshold,
            max_expand=self.max_expand,
            min_window=self.min_window,
        )
        return expander.postprocess_nodes(nodes, QueryBundle(query_str=query))
