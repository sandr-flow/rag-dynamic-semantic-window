"""Retrieval strategies for benchmark comparison."""

from abc import ABC, abstractmethod

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (
    CodeSplitter,
    HTMLNodeParser,
    JSONNodeParser,
    MarkdownNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document, NodeWithScore, QueryBundle

from src.config import (
    DEFAULT_CODE_SPLITTER_CONFIG,
    DEFAULT_DYNAMIC_SEMANTIC_CONFIG,
    DEFAULT_EXPANSION_CONFIG,
    DEFAULT_FIXED_WINDOW_CONFIG,
    DEFAULT_HTML_SPLITTER_CONFIG,
    DEFAULT_JSON_SPLITTER_CONFIG,
    DEFAULT_MARKDOWN_SPLITTER_CONFIG,
    DEFAULT_NAIVE_CHUNKING_CONFIG,
    DEFAULT_RETRIEVAL_CONFIG,
    DEFAULT_SEMANTIC_SPLITTER_CONFIG,
    DEFAULT_TOKEN_TEXT_SPLITTER_CONFIG,
    CodeSplitterConfig,
    DynamicSemanticConfig,
    ExpansionConfig,
    FixedWindowConfig,
    HTMLSplitterConfig,
    JSONSplitterConfig,
    MarkdownSplitterConfig,
    NaiveChunkingConfig,
    SemanticSplitterConfig,
    TokenTextSplitterConfig,
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


class MarkdownSplitterStrategy(BaseStrategy):
    """Baseline strategy using LlamaIndex MarkdownNodeParser."""

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        config: MarkdownSplitterConfig | None = None,
    ):
        self.config = config or DEFAULT_MARKDOWN_SPLITTER_CONFIG
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Markdown Splitter"

    def _build_index(self) -> None:
        parser = MarkdownNodeParser(header_path_separator=self.config.header_path_separator)
        nodes = parser.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        return retriever.retrieve(query)


class HTMLSplitterStrategy(BaseStrategy):
    """Baseline strategy using LlamaIndex HTMLNodeParser."""

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        config: HTMLSplitterConfig | None = None,
    ):
        self.config = config or DEFAULT_HTML_SPLITTER_CONFIG
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "HTML Splitter"

    def _build_index(self) -> None:
        parser = HTMLNodeParser(tags=list(self.config.tags))
        nodes = parser.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        return retriever.retrieve(query)


class JSONSplitterStrategy(BaseStrategy):
    """Baseline strategy using LlamaIndex JSONNodeParser."""

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        config: JSONSplitterConfig | None = None,
    ):
        self.config = config or DEFAULT_JSON_SPLITTER_CONFIG
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "JSON Splitter"

    def _build_index(self) -> None:
        parser = JSONNodeParser(include_metadata=self.config.include_metadata)
        nodes = parser.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(nodes)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        return retriever.retrieve(query)


class CodeSplitterStrategy(BaseStrategy):
    """Baseline strategy using LlamaIndex CodeSplitter."""

    def __init__(
        self,
        documents: list[Document],
        top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
        config: CodeSplitterConfig | None = None,
    ):
        self.config = config or DEFAULT_CODE_SPLITTER_CONFIG
        super().__init__(documents, top_k)

    @property
    def name(self) -> str:
        return "Code Splitter"

    def _build_index(self) -> None:
        splitter = CodeSplitter(
            language=self.config.language,
            chunk_lines=self.config.chunk_lines,
            chunk_lines_overlap=self.config.chunk_lines_overlap,
            max_chars=self.config.max_chars,
            count_mode=self.config.count_mode,
            max_tokens=self.config.max_tokens,
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
        seed_rejection_log_path: str | None = None,
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
            seed_rejection_log_path: Path to log rejected seeds (JSONL format).
            phantom_window: Number of neighbors to include in embedding context (0 = disabled).
            prefetch_multiplier: Multiplier for first-pass retrieval (4 = fetch top_k*4 seeds).
        """
        self.seed_rejection_log_path = seed_rejection_log_path
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

        # Apply dynamic expansion - uses defaults from config.py
        expander = DynamicSemanticExpander(
            docstore=self.docstore,
            seed_rejection_log_path=self.seed_rejection_log_path,
            target_clusters=self.top_k,
            threshold=self.expansion_config.threshold,
            skip_threshold=self.expansion_config.skip_threshold,
            max_expand=self.expansion_config.max_expand,
            min_window=self.expansion_config.min_window,
            min_chunk_length=self.expansion_config.min_chunk_length,
            relevance_threshold_pct=self.expansion_config.relevance_threshold_pct,
            merge_gap=self.expansion_config.merge_gap,
        )
        return expander.postprocess_nodes(nodes, QueryBundle(query_str=query))


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



