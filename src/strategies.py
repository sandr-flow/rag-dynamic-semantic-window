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


def _interleave_node_lists(
    primary: list[NodeWithScore], secondary: list[NodeWithScore]
) -> list[NodeWithScore]:
    """Rank-interleave two seed lists, keeping the first copy of each node id.

    A duplicate is skipped and the next unused node from that same list fills
    the slot, matching :func:`src.seed_retrieval.interleave_ranked_indices`.
    """
    seen: set[str] = set()
    merged: list[NodeWithScore] = []
    buckets = (primary, secondary)
    pointers = [0, 0]
    while True:
        progressed = False
        for i, bucket in enumerate(buckets):
            while pointers[i] < len(bucket):
                item = bucket[pointers[i]]
                pointers[i] += 1
                node_id = item.node.node_id
                if node_id in seen:
                    continue
                seen.add(node_id)
                merged.append(item)
                progressed = True
                break
        if not progressed:
            break
    return merged


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
        self._matrix: np.ndarray | None = None
        self._matrix_node_ids: list[str] | None = None
        self._clean_matrix: np.ndarray | None = None
        self._clean_node_ids: list[str] | None = None
        self._build_index()

    def _ensure_matrix(self) -> None:
        """Materialize all index embeddings into one normalized numpy matrix.

        ``SimpleVectorStore.query`` rebuilds a numpy array from a python list
        of lists on *every* query, which is unbearably slow on large shared
        indexes. Doing the conversion once turns each query into a single
        matrix-vector product.
        """
        if self._matrix is not None:
            return
        data = self.index.vector_store._data
        node_ids = list(data.embedding_dict.keys())
        matrix = np.asarray(
            [data.embedding_dict[nid] for nid in node_ids], dtype=np.float32
        )
        self._matrix = self._normalize_rows(matrix)
        self._matrix_node_ids = node_ids

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _normalize_query(self, query: str) -> np.ndarray:
        q = np.asarray(Settings.embed_model.get_query_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm:
            q = q / q_norm
        return q

    def _topk_from_matrix(
        self,
        matrix: np.ndarray,
        node_ids: list[str],
        query_vec: np.ndarray,
        top_k: int,
    ) -> list[NodeWithScore]:
        scores = matrix @ query_vec
        k = min(top_k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        docstore = self.index.docstore
        return [
            NodeWithScore(
                node=docstore.get_node(node_ids[i]),
                score=float(scores[i]),
            )
            for i in top
        ]

    def _fast_retrieve(self, query: str, top_k: int) -> list[NodeWithScore]:
        """Cosine top-k over the precomputed matrix (SimpleVectorStore parity)."""
        self._ensure_matrix()
        return self._topk_from_matrix(
            self._matrix, self._matrix_node_ids, self._normalize_query(query), top_k
        )

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
        return self._fast_retrieve(query, self.top_k)


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
        nodes = self._fast_retrieve(query, self.top_k)

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
        return self._fast_retrieve(query, self.top_k)


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
        if adjacency_matrix is not None:
            self._clean_matrix = self._normalize_rows(
                np.asarray(adjacency_matrix, dtype=np.float32)
            )
            self._clean_node_ids = [node.node_id for node in nodes]
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
        When dual-seed is on, the same budget is also taken from clean sentence
        embeddings (Fixed Window matching) and the two lists are interleaved.
        Second pass: Expand and deduplicate to target top_k results.
        """
        prefetch_k = self.top_k * self.prefetch_multiplier
        self._ensure_matrix()
        query_vec = self._normalize_query(query)
        phantom_nodes = self._topk_from_matrix(
            self._matrix, self._matrix_node_ids, query_vec, prefetch_k
        )
        nodes = phantom_nodes
        if (
            self.dynamic_config.dual_seed
            and self._clean_matrix is not None
            and self._clean_node_ids is not None
        ):
            clean_nodes = self._topk_from_matrix(
                self._clean_matrix, self._clean_node_ids, query_vec, prefetch_k
            )
            nodes = _interleave_node_lists(phantom_nodes, clean_nodes)

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
        return self._fast_retrieve(query, self.top_k)



