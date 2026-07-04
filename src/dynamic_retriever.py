"""LlamaIndex adapter for the Dynamic Semantic expansion core.

All expansion logic lives in :mod:`src.expansion_core`. This module only
normalizes the per-sentence arrays handed over by the strategy, maps
retrieved seed nodes to sentence positions, invokes the core, and turns the
resulting sentence spans back into LlamaIndex nodes.

The adapter deliberately takes sentence texts and embeddings directly
instead of reading them from a docstore: ``VectorStoreIndex`` overwrites
docstore entries with embedding-stripped copies, so a docstore is not a
reliable embedding source.
"""

import numpy as np
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pydantic import ConfigDict, PrivateAttr

from src.config import DEFAULT_ADAPTIVE_THRESHOLD_CONFIG, DEFAULT_EXPANSION_CONFIG
from src.expansion_core import DynamicExpansionCore, build_garbage_mask


class DynamicSemanticExpander(BaseNodePostprocessor):
    """
    Expand retrieved sentence nodes into semantically coherent windows.

    Thin adapter over :class:`DynamicExpansionCore`: normalizes the provided
    sentence embeddings and precomputes adjacent similarities once, then per
    query maps retrieved seeds to positions, computes query-sentence
    similarities, and reconstructs the merged spans returned by the core as
    new ``TextNode`` objects.

    Attributes:
        sentences: Sentence texts in document order.
        node_ids: Node ids aligned with ``sentences`` (used to map retrieved
            seeds to positions and to report ``source_nodes`` metadata).
        embeddings: Sentence embeddings aligned with ``sentences``, shape
            (num_sentences, dim). Normalized internally.
        query_embedding: Optional externally computed query embedding; when
            unset, the query from ``query_bundle`` is embedded via
            ``Settings.embed_model``. Without either, expansion runs without
            query-aware checks and no backfill beyond the retrieved seeds.
    """

    sentences: list[str]
    node_ids: list[str]
    embeddings: np.ndarray
    doc_ids: list[str] | None = None

    # Expansion parameters (see ExpansionConfig)
    threshold: float = DEFAULT_EXPANSION_CONFIG.threshold
    skip_threshold: float = DEFAULT_EXPANSION_CONFIG.skip_threshold
    max_expand: int = DEFAULT_EXPANSION_CONFIG.max_expand
    min_window: int = DEFAULT_EXPANSION_CONFIG.min_window
    min_chunk_length: int = DEFAULT_EXPANSION_CONFIG.min_chunk_length
    relevance_threshold_pct: float = DEFAULT_EXPANSION_CONFIG.relevance_threshold_pct
    merge_gap: int = DEFAULT_EXPANSION_CONFIG.merge_gap
    target_clusters: int = DEFAULT_EXPANSION_CONFIG.target_clusters

    # Adaptive threshold parameters (see AdaptiveThresholdConfig)
    adaptive_threshold_enabled: bool = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.enabled
    decay_lambda_sparse: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.decay_lambda_sparse
    decay_lambda_dense: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.decay_lambda_dense
    density_threshold: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.density_threshold
    density_score_ratio: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.density_score_ratio
    floor_multiplier: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.floor_multiplier
    gradient_cliff_factor: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.gradient_cliff_factor

    query_embedding: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Derived arrays, built lazily once
    _prepared: bool = PrivateAttr(default=False)
    _normalized: np.ndarray | None = PrivateAttr(default=None)
    _neighbor_sims: np.ndarray | None = PrivateAttr(default=None)
    _garbage_mask: np.ndarray | None = PrivateAttr(default=None)
    _id_to_pos: dict = PrivateAttr(default_factory=dict)
    _segment_ids: np.ndarray | None = PrivateAttr(default=None)

    def _ensure_prepared(self) -> None:
        """Normalize embeddings and precompute derived arrays once."""
        if self._prepared:
            return

        embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != len(self.sentences):
            raise ValueError(
                "embeddings must be a (num_sentences, dim) array aligned with sentences"
            )
        if self.doc_ids is not None and len(self.doc_ids) != len(self.sentences):
            raise ValueError("doc_ids must be aligned with sentences")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        self._normalized = normalized
        self._neighbor_sims = (
            np.sum(normalized[:-1] * normalized[1:], axis=1)
            if len(normalized) > 1
            else np.array([], dtype=np.float32)
        )
        self._garbage_mask = build_garbage_mask(self.sentences, self.min_chunk_length)
        self._id_to_pos = {node_id: pos for pos, node_id in enumerate(self.node_ids)}
        self._segment_ids = np.asarray(
            self.doc_ids if self.doc_ids is not None else ["doc"] * len(self.sentences),
            dtype=object,
        )
        self._prepared = True

    def _query_sims(self, query_bundle: QueryBundle | None) -> np.ndarray | None:
        """Query-sentence similarities, or None when no query is available."""
        query = self.query_embedding
        if query is None and query_bundle is not None and query_bundle.query_str:
            from llama_index.core import Settings

            query = np.array(
                Settings.embed_model.get_text_embedding(query_bundle.query_str),
                dtype=np.float32,
            )
        if query is None:
            return None

        query = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        return self._normalized @ query

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Map retrieved seeds to positions, run the core, rebuild nodes."""
        if not nodes:
            return []

        self._ensure_prepared()
        num_sentences = len(self.sentences)
        if num_sentences == 0:
            return []

        sentence_sims = self._query_sims(query_bundle)
        query_aware = sentence_sims is not None
        if sentence_sims is None:
            sentence_sims = np.zeros(num_sentences, dtype=np.float32)

        # Candidates: retrieved seeds in rank order; with a query, extend with
        # remaining sentences by query similarity so the core can backfill up
        # to target_clusters.
        seen: set[int] = set()
        candidates: list[int] = []
        for node_score in nodes:
            pos = self._id_to_pos.get(node_score.node.node_id)
            if pos is None or pos in seen:
                continue
            seen.add(pos)
            candidates.append(pos)
        if not candidates:
            return []
        if query_aware:
            for pos in np.argsort(sentence_sims)[::-1]:
                pos = int(pos)
                if pos not in seen:
                    candidates.append(pos)

        core = DynamicExpansionCore(
            neighbor_sims=self._neighbor_sims,
            sentence_sims=sentence_sims,
            top_k_indices=np.array(candidates, dtype=np.int32),
            threshold=self.threshold,
            skip_threshold=self.skip_threshold,
            min_window=self.min_window,
            max_expand=self.max_expand,
            relevance_threshold_pct=self.relevance_threshold_pct,
            merge_gap=self.merge_gap,
            target_clusters=self.target_clusters,
            adaptive_threshold_enabled=self.adaptive_threshold_enabled,
            decay_lambda_sparse=self.decay_lambda_sparse,
            decay_lambda_dense=self.decay_lambda_dense,
            density_threshold=self.density_threshold,
            density_score_ratio=self.density_score_ratio,
            floor_multiplier=self.floor_multiplier,
            gradient_cliff_factor=self.gradient_cliff_factor,
            query_aware_enabled=query_aware,
            garbage_mask=self._garbage_mask,
            segment_ids=self._segment_ids,
        )
        clusters = core.expand_and_retrieve()

        results = []
        for cluster in clusters:
            span = range(cluster.start_idx, cluster.end_idx + 1)
            source_doc = str(self._segment_ids[cluster.seed_idx])
            merged_text = " ".join(self.sentences[i] for i in span)
            new_node = TextNode(
                text=merged_text,
                metadata={
                    "source_nodes": [self.node_ids[i] for i in span],
                    "source_doc": source_doc,
                    "expansion_size": len(span),
                    "seed_index": cluster.seed_idx,
                },
            )
            results.append(NodeWithScore(node=new_node, score=cluster.score))
        return results
