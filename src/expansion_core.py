"""Single implementation of the Dynamic Semantic expansion algorithm.

This is the only place where expansion/merge/backfill logic lives. Both entry
points share it:

- benchmark path: ``DynamicSemanticExpander`` (src/dynamic_retriever.py)
  assembles arrays from a LlamaIndex docstore and maps spans back to nodes;
- HPO path: the Optuna objective (run_optuna.py) feeds pre-computed corpus
  arrays directly.

The core is pure: numpy arrays in, index spans out. No I/O, no LlamaIndex.

Resolved divergences between the two historic implementations
(``DynamicSemanticExpander`` vs ``CachedDynamicExpander``):

- adaptive threshold and gradient stop (previously benchmark-path only) are
  part of the core, so they now also apply during HPO;
- skip ("bridge") similarity uses adjacent-pair sims — ``sim(skipped,
  beyond)`` — rather than the old benchmark path's two-step ``sim(current,
  beyond)``, which cannot be derived from adjacency arrays;
- merged clusters are returned sorted by seed score;
- garbage sentences (too short / reference sections / headings) trim cluster
  edges via an optional precomputed mask instead of post-hoc node filtering;
- section headers are jumped during expansion so a heading cliff does not
  stop the window short of the next content sentence;
- seed validation (Modified Z-score) was removed: it was disabled by default
  and rejected valid seeds (see docs/math_foundations.md, section A).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from src.config import DEFAULT_ADAPTIVE_THRESHOLD_CONFIG, DEFAULT_EXPANSION_CONFIG
from src.tokens import count_tokens

# Sentences that must not appear at cluster edges (references, links, etc.)
GARBAGE_PATTERNS = re.compile(
    r"^\s*(References|See also|External links|Notes|Bibliography|Further reading)",
    re.IGNORECASE,
)

# Consecutive section titles jumped without spending an expand step.
MAX_HEADER_SKIP = 3

# Numbered / lettered prefixes: "3.", "3.1.2", "3 Methodology", "IV.", "A."
_HEADER_NUMBERING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)[.)]?|(?:[IVXLCM]{1,8}|[A-Z])[.)])\s+",
)

_FIGURE_TABLE = re.compile(
    r"^(?:figure|table|eq(?:uation)?|fig\.|tab\.)\b",
    re.IGNORECASE,
)

# Common IMRaD / wiki headings. Unknown Title-Case headings (2+ words) are
# still detected below; this set covers single-word titles like "Methodology".
_SECTION_TITLES = frozenset(
    {
        "abstract",
        "acknowledgements",
        "acknowledgments",
        "analysis",
        "appendix",
        "approach",
        "background",
        "bibliography",
        "broader impact",
        "case study",
        "conclusion",
        "conclusions",
        "dataset",
        "datasets",
        "discussion",
        "error analysis",
        "ethical considerations",
        "evaluation",
        "experiment",
        "experimental results",
        "experimental setup",
        "experiments",
        "external links",
        "further reading",
        "future work",
        "implementation",
        "inference",
        "introduction",
        "limitations",
        "method",
        "methodology",
        "methods",
        "model",
        "notation",
        "notes",
        "our approach",
        "overview",
        "preliminaries",
        "problem formulation",
        "problem statement",
        "proposed approach",
        "proposed method",
        "qualitative analysis",
        "references",
        "related work",
        "related works",
        "results",
        "results and discussion",
        "see also",
        "setup",
        "task definition",
        "training",
    }
)


def is_section_header(text: str) -> bool:
    """True for heading-like lines (IMRaD titles, numbered sections, Title Case)."""
    raw = text.strip()
    if not raw or len(raw) > 80:
        return False
    stripped = re.sub(r"^[=#*]+\s*|\s*[=#*]+$", "", raw).strip().rstrip(":")
    if not stripped or _FIGURE_TABLE.match(stripped):
        return False
    body = _HEADER_NUMBERING.sub("", stripped, count=1).strip()
    if not body or body[-1] in ".?!":
        return False
    words = body.split()
    if not 1 <= len(words) <= 8:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 ,\-/&'()]*", body):
        return False
    key = re.sub(r"\s+", " ", body.lower())
    if key in _SECTION_TITLES:
        return True
    if body.isupper() and len(body) >= 4:
        return True
    if len(words) >= 2 and all(word[0].isupper() for word in words if word[0].isalpha()):
        return True
    return False


def build_header_mask(sentences: list[str]) -> np.ndarray:
    """Boolean mask of section-header sentences (True = jump during expansion)."""
    return np.array([is_section_header(sentence) for sentence in sentences], dtype=bool)


@dataclass
class ExpandedCluster:
    """Result of cluster expansion.

    Attributes:
        start_idx: Starting sentence index (inclusive).
        end_idx: Ending sentence index (inclusive).
        seed_idx: Index of the seed sentence.
        score: Seed's similarity score to the query.
    """

    start_idx: int
    end_idx: int
    seed_idx: int
    score: float


def build_garbage_mask(
    sentences: list[str],
    min_chunk_length: int = DEFAULT_EXPANSION_CONFIG.min_chunk_length,
) -> np.ndarray:
    """Boolean mask of garbage sentences (True = trim from cluster edges)."""
    mask = np.zeros(len(sentences), dtype=bool)
    for i, sentence in enumerate(sentences):
        text = sentence.strip()
        if (
            len(text) < min_chunk_length
            or GARBAGE_PATTERNS.match(text)
            or is_section_header(text)
        ):
            mask[i] = True
    return mask


class DynamicExpansionCore:
    """
    Dynamic Semantic expansion over pre-computed similarity arrays.

    All similarity lookups are O(1) numpy accesses:
    - ``neighbor_sims[i]`` = cos(sentence_i, sentence_{i+1})
    - ``sentence_sims[i]`` = cos(query, sentence_i)

    Expansion uses a HYBRID WINDOW approach:
    - always include ±min_window neighbors (safety net);
    - expand further while adjacency similarity clears the (optionally
      adaptive, distance-decayed) threshold;
    - query-aware fallback: a neighbor failing the adjacency check is still
      included when it is directly relevant to the query;
    - skip logic bridges one weak sentence when the pair beyond it is strong;
    - section headers are jumped without spending an expand step; the first
      content sentence after a heading is taken without an adjacency check
      (the heading-to-content pair is a false cliff);
    - gradient stop halts on a semantic cliff (sharp acceleration of the
      similarity drop) when the adaptive threshold is enabled.

    Args:
        neighbor_sims: Adjacent sentence similarities, shape (n-1,).
        sentence_sims: Query-sentence similarities, shape (n,).
        top_k_indices: Candidate seed indices in priority order. Entries
            beyond the initial seeds are consumed by backfill until
            ``target_clusters`` clusters exist.
        threshold: Base adjacency threshold beyond min_window.
        skip_threshold: Threshold for bridging a single weak sentence.
        min_window: Neighbors to always include on each side.
        max_expand: Maximum expansion steps in each direction.
        relevance_threshold_pct: Query-relevance fallback threshold as a
            fraction of the best candidate's score.
        merge_gap: Maximum gap between clusters to merge.
        target_clusters: Number of output clusters to aim for.
        adaptive_threshold_enabled: Distance-decay the threshold and enable
            the gradient stop.
        query_aware_enabled: Disable to ignore ``sentence_sims`` during
            expansion (used when no query embedding is available); the
            adaptive decay then anchors on a neutral seed score of 1.0.
        garbage_mask: Optional boolean mask (True = garbage); garbage
            sentences are trimmed from cluster edges, empty clusters dropped.
        header_mask: Optional boolean mask (True = section header). Headers
            are jumped during expansion and trimmed from cluster edges.
        segment_ids: Optional segment/document id per sentence. Expansion and
            merge never cross segment boundaries.
    """

    def __init__(
        self,
        neighbor_sims: np.ndarray,
        sentence_sims: np.ndarray,
        top_k_indices: np.ndarray,
        threshold: float = DEFAULT_EXPANSION_CONFIG.threshold,
        skip_threshold: float = DEFAULT_EXPANSION_CONFIG.skip_threshold,
        min_window: int = DEFAULT_EXPANSION_CONFIG.min_window,
        max_expand: int = DEFAULT_EXPANSION_CONFIG.max_expand,
        relevance_threshold_pct: float = DEFAULT_EXPANSION_CONFIG.relevance_threshold_pct,
        merge_gap: int = DEFAULT_EXPANSION_CONFIG.merge_gap,
        target_clusters: int = DEFAULT_EXPANSION_CONFIG.target_clusters,
        adaptive_threshold_enabled: bool = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.enabled,
        decay_lambda_sparse: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.decay_lambda_sparse,
        decay_lambda_dense: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.decay_lambda_dense,
        density_threshold: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.density_threshold,
        density_score_ratio: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.density_score_ratio,
        floor_multiplier: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.floor_multiplier,
        gradient_cliff_factor: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.gradient_cliff_factor,
        query_aware_enabled: bool = True,
        garbage_mask: np.ndarray | None = None,
        header_mask: np.ndarray | None = None,
        segment_ids: np.ndarray | list[str] | None = None,
    ):
        self.neighbor_sims = neighbor_sims
        self.sentence_sims = sentence_sims
        self.top_k_indices = top_k_indices
        self.num_sentences = len(sentence_sims)

        self.threshold = threshold
        self.skip_threshold = skip_threshold
        self.min_window = min_window
        self.max_expand = max_expand
        self.relevance_threshold_pct = relevance_threshold_pct
        self.merge_gap = merge_gap
        self.target_clusters = target_clusters

        self.adaptive_threshold_enabled = adaptive_threshold_enabled
        self.decay_lambda_sparse = decay_lambda_sparse
        self.decay_lambda_dense = decay_lambda_dense
        self.density_threshold = density_threshold
        self.density_score_ratio = density_score_ratio
        self.floor_multiplier = floor_multiplier
        self.gradient_cliff_factor = gradient_cliff_factor

        self.query_aware_enabled = query_aware_enabled
        self.garbage_mask = garbage_mask
        self.header_mask = header_mask
        if self.header_mask is not None and len(self.header_mask) != self.num_sentences:
            raise ValueError("header_mask must be aligned with sentence_sims")
        self.segment_ids = (
            np.asarray(segment_ids, dtype=object) if segment_ids is not None else None
        )
        if self.segment_ids is not None and len(self.segment_ids) != self.num_sentences:
            raise ValueError("segment_ids must be aligned with sentence_sims")

    def expand_and_retrieve(self) -> list[ExpandedCluster]:
        """
        Expand clusters from candidates with iterative backfill.

        Guarantees ``target_clusters`` output (if enough candidates exist) by:
        1. Process a batch of candidates.
        2. Expand + merge.
        3. If < target_clusters, take further uncovered candidates and repeat.

        Returns:
            List of ExpandedCluster objects, sorted by seed score descending.
        """
        if len(self.top_k_indices) == 0:
            return []

        # Relevance threshold anchors on the best candidate's score
        max_seed_score = self.sentence_sims[self.top_k_indices[0]]
        relevance_threshold = max_seed_score * self.relevance_threshold_pct

        processed_candidates: set[int] = set()
        covered_indices: set[int] = set()
        final_clusters: list[ExpandedCluster] = []
        candidate_ptr = 0

        while (
            len(final_clusters) < self.target_clusters
            and candidate_ptr < len(self.top_k_indices)
        ):
            # Collect a batch of new candidates not yet covered
            new_seeds: list[int] = []
            while (
                len(new_seeds) < self.target_clusters
                and candidate_ptr < len(self.top_k_indices)
            ):
                candidate_idx = int(self.top_k_indices[candidate_ptr])
                candidate_ptr += 1
                if candidate_idx in processed_candidates or candidate_idx in covered_indices:
                    continue
                processed_candidates.add(candidate_idx)
                new_seeds.append(candidate_idx)

            if not new_seeds:
                break

            new_clusters: list[ExpandedCluster] = []
            for seed_idx in new_seeds:
                cluster = self._trim_garbage(
                    self._expand_cluster(seed_idx, relevance_threshold)
                )
                if cluster:
                    new_clusters.append(cluster)

            if not new_clusters:
                continue

            final_clusters = self._merge_clusters(final_clusters + new_clusters)
            covered_indices = self.get_all_retrieved_indices(final_clusters)

        return final_clusters[: self.target_clusters]

    def _expand_cluster(
        self,
        seed_idx: int,
        relevance_threshold: float,
    ) -> ExpandedCluster | None:
        """Expand context around a single seed using the hybrid window."""
        if seed_idx < 0 or seed_idx >= self.num_sentences:
            return None

        # Anchor for adaptive decay; neutral when no query is available.
        seed_score = (
            float(self.sentence_sims[seed_idx]) if self.query_aware_enabled else 1.0
        )

        # Expand left
        left_idx = seed_idx
        left_scores: list[float] = []
        for i in range(self.max_expand):
            left_idx, jumped_header = self._jump_headers(seed_idx, left_idx, -1)
            next_left = left_idx - 1
            if next_left < 0 or not self._same_segment(seed_idx, next_left):
                break

            if i >= self.min_window and not jumped_header:
                adj_sim = (
                    float(self.neighbor_sims[next_left])
                    if next_left < len(self.neighbor_sims)
                    else 0.0
                )
                left_scores.append(adj_sim)
                if self._should_stop_gradient(left_scores):
                    break

                if adj_sim < self._adaptive_threshold(seed_score, left_scores, i + 1):
                    query_relevant = (
                        self.query_aware_enabled
                        and self.sentence_sims[next_left] >= relevance_threshold
                    )
                    if not query_relevant:
                        # Bridge one weak sentence if the pair beyond is strong
                        if next_left - 1 >= 0 and self._same_segment(
                            seed_idx, next_left - 1
                        ):
                            skip_sim = (
                                float(self.neighbor_sims[next_left - 1])
                                if next_left - 1 < len(self.neighbor_sims)
                                else 0.0
                            )
                            if skip_sim >= self.skip_threshold:
                                left_scores.append(skip_sim)
                                left_idx = next_left - 1
                                continue
                        break

            left_idx = next_left

        # Expand right
        right_idx = seed_idx
        right_scores: list[float] = []
        for i in range(self.max_expand):
            right_idx, jumped_header = self._jump_headers(seed_idx, right_idx, 1)
            next_right = right_idx + 1
            if next_right >= self.num_sentences or not self._same_segment(
                seed_idx, next_right
            ):
                break

            if i >= self.min_window and not jumped_header:
                adj_sim = (
                    float(self.neighbor_sims[right_idx])
                    if right_idx < len(self.neighbor_sims)
                    else 0.0
                )
                right_scores.append(adj_sim)
                if self._should_stop_gradient(right_scores):
                    break

                if adj_sim < self._adaptive_threshold(seed_score, right_scores, i + 1):
                    query_relevant = (
                        self.query_aware_enabled
                        and self.sentence_sims[next_right] >= relevance_threshold
                    )
                    if not query_relevant:
                        # Bridge one weak sentence if the pair beyond is strong
                        if (
                            next_right + 1 < self.num_sentences
                            and self._same_segment(seed_idx, next_right + 1)
                            and next_right < len(self.neighbor_sims)
                        ):
                            skip_sim = float(self.neighbor_sims[next_right])
                            if skip_sim >= self.skip_threshold:
                                right_scores.append(skip_sim)
                                right_idx = next_right + 1
                                continue
                        break

            right_idx = next_right

        return ExpandedCluster(
            start_idx=left_idx,
            end_idx=right_idx,
            seed_idx=seed_idx,
            score=float(self.sentence_sims[seed_idx]),
        )

    def _same_segment(self, seed_idx: int, candidate_idx: int) -> bool:
        """Return True when two sentence positions belong to one segment."""
        if self.segment_ids is None:
            return True
        return self.segment_ids[seed_idx] == self.segment_ids[candidate_idx]

    def _is_header(self, idx: int) -> bool:
        return self.header_mask is not None and bool(self.header_mask[idx])

    def _is_edge_noise(self, idx: int) -> bool:
        if self.garbage_mask is not None and bool(self.garbage_mask[idx]):
            return True
        return self._is_header(idx)

    def _jump_headers(self, seed_idx: int, idx: int, step: int) -> tuple[int, bool]:
        """Include consecutive section titles in the span without spending a step."""
        jumped = False
        n_skipped = 0
        while n_skipped < MAX_HEADER_SKIP:
            nxt = idx + step
            if nxt < 0 or nxt >= self.num_sentences:
                break
            if not self._same_segment(seed_idx, nxt) or not self._is_header(nxt):
                break
            idx = nxt
            jumped = True
            n_skipped += 1
        return idx, jumped

    def _adaptive_threshold(
        self,
        seed_score: float,
        local_scores: list[float],
        distance: int,
    ) -> float:
        """
        Expansion threshold decayed with distance, adjusted by local density.

        In dense regions (many high adjacency scores) decay is slower to
        capture more context. Floored at ``threshold * floor_multiplier``.
        """
        if not self.adaptive_threshold_enabled or not local_scores:
            return self.threshold

        high_score_count = sum(
            1 for s in local_scores if s > seed_score * self.density_score_ratio
        )
        density = high_score_count / len(local_scores)
        decay_lambda = (
            self.decay_lambda_dense
            if density > self.density_threshold
            else self.decay_lambda_sparse
        )
        decay_threshold = seed_score * math.exp(-decay_lambda * distance)
        return max(decay_threshold, self.threshold * self.floor_multiplier)

    def _should_stop_gradient(self, scores_sequence: list[float]) -> bool:
        """
        Detect a semantic cliff via gradient acceleration.

        Stops expansion when the similarity drop accelerates beyond
        ``gradient_cliff_factor``, indicating a crossed topic boundary.
        """
        if not self.adaptive_threshold_enabled or len(scores_sequence) < 3:
            return False

        gradients = np.diff(scores_sequence)
        if len(gradients) < 2:
            return False

        prev_gradient = gradients[-2] if abs(gradients[-2]) > 1e-10 else 1e-10
        return gradients[-1] / prev_gradient < self.gradient_cliff_factor

    def _trim_garbage(self, cluster: ExpandedCluster | None) -> ExpandedCluster | None:
        """Trim garbage/header sentences from cluster edges; drop empty clusters."""
        if cluster is None:
            return cluster
        if self.garbage_mask is None and self.header_mask is None:
            return cluster

        start, end = cluster.start_idx, cluster.end_idx
        while start <= end and self._is_edge_noise(start):
            start += 1
        while end >= start and self._is_edge_noise(end):
            end -= 1
        if start > end:
            return None

        cluster.start_idx = start
        cluster.end_idx = end
        return cluster

    def _merge_clusters(self, clusters: list[ExpandedCluster]) -> list[ExpandedCluster]:
        """
        Merge overlapping or adjacent clusters.

        Clusters merge when they overlap or lie within ``merge_gap`` of each
        other. Returns clusters sorted by score descending.
        """
        if not clusters:
            return []

        clusters = sorted(clusters, key=lambda c: c.score, reverse=True)

        merged: list[ExpandedCluster] = []
        used = [False] * len(clusters)

        for i, cluster_i in enumerate(clusters):
            if used[i]:
                continue

            merged_start = cluster_i.start_idx
            merged_end = cluster_i.end_idx
            best_score = cluster_i.score
            best_seed = cluster_i.seed_idx
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j, cluster_j in enumerate(clusters):
                    if used[j]:
                        continue
                    if (
                        self._same_segment(cluster_i.seed_idx, cluster_j.seed_idx)
                        and cluster_j.start_idx <= merged_end + self.merge_gap
                        and cluster_j.end_idx >= merged_start - self.merge_gap
                    ):
                        merged_start = min(merged_start, cluster_j.start_idx)
                        merged_end = max(merged_end, cluster_j.end_idx)
                        if cluster_j.score > best_score:
                            best_score = cluster_j.score
                            best_seed = cluster_j.seed_idx
                        used[j] = True
                        changed = True

            merged.append(
                ExpandedCluster(
                    start_idx=merged_start,
                    end_idx=merged_end,
                    seed_idx=best_seed,
                    score=best_score,
                )
            )

        return sorted(merged, key=lambda c: c.score, reverse=True)

    def get_cluster_sentence_indices(self, cluster: ExpandedCluster) -> list[int]:
        """Get list of sentence indices in a cluster."""
        return list(range(cluster.start_idx, cluster.end_idx + 1))

    def get_all_retrieved_indices(self, clusters: list[ExpandedCluster]) -> set[int]:
        """Get all unique sentence indices across all clusters."""
        indices: set[int] = set()
        for cluster in clusters:
            indices.update(range(cluster.start_idx, cluster.end_idx + 1))
        return indices


def evaluate_retrieval(
    clusters: list[ExpandedCluster],
    answer_sentence_idx: int,
    sentences: list[str],
) -> dict:
    """
    Evaluate retrieval quality for a single question.

    Args:
        clusters: List of expanded clusters.
        answer_sentence_idx: Index of the answer sentence (-1 if not found).
        sentences: List of all sentences.

    Returns:
        Dictionary with metrics: hit (bool), tokens (int), num_clusters, rank.
    """
    if answer_sentence_idx < 0:
        return {"hit": False, "tokens": 0, "num_clusters": len(clusters), "rank": -1}

    hit = False
    rank = -1
    for i, cluster in enumerate(clusters):
        if cluster.start_idx <= answer_sentence_idx <= cluster.end_idx:
            hit = True
            rank = i + 1  # 1-indexed rank
            break

    # Token accounting mirrors the benchmark path: sentences of a cluster are
    # joined with spaces into one chunk, chunks are joined with spaces.
    cluster_texts = []
    for cluster in clusters:
        parts = [
            sentences[idx]
            for idx in range(cluster.start_idx, cluster.end_idx + 1)
            if 0 <= idx < len(sentences)
        ]
        if parts:
            cluster_texts.append(" ".join(parts))
    tokens = count_tokens(" ".join(cluster_texts))

    return {
        "hit": hit,
        "tokens": tokens,
        "num_clusters": len(clusters),
        "rank": rank,
    }


def evaluate_multi_retrieval(
    clusters: list[ExpandedCluster],
    answer_sentence_indices: list[int],
    sentences: list[str],
) -> dict:
    """Evaluate joint multi-answer retrieval (extrahard / shared index)."""
    if not answer_sentence_indices or any(idx < 0 for idx in answer_sentence_indices):
        return {"hit": False, "tokens": 0, "num_clusters": len(clusters), "rank": -1, "mrr": 0.0}

    ranks: list[float] = []
    hit = True
    for answer_idx in answer_sentence_indices:
        rank = -1
        for i, cluster in enumerate(clusters):
            if cluster.start_idx <= answer_idx <= cluster.end_idx:
                rank = i + 1
                break
        if rank < 0:
            hit = False
            ranks.append(0.0)
        else:
            ranks.append(1.0 / rank)

    cluster_texts = []
    for cluster in clusters:
        parts = [
            sentences[idx]
            for idx in range(cluster.start_idx, cluster.end_idx + 1)
            if 0 <= idx < len(sentences)
        ]
        if parts:
            cluster_texts.append(" ".join(parts))
    tokens = count_tokens(" ".join(cluster_texts))

    return {
        "hit": hit,
        "tokens": tokens,
        "num_clusters": len(clusters),
        "rank": int(ranks[0] > 0) if ranks else -1,
        "mrr": sum(ranks) / len(ranks),
    }
