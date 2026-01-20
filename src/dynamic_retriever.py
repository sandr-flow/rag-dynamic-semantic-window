"""Dynamic Semantic Expander - Custom NodePostprocessor for context expansion."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore import BaseDocumentStore

from src.config import (
    DEFAULT_ADAPTIVE_THRESHOLD_CONFIG,
    DEFAULT_EXPANSION_CONFIG,
    DEFAULT_SEED_VALIDATION_CONFIG,
)


# Patterns to filter garbage chunks (references, links, etc.)
GARBAGE_PATTERNS = re.compile(
    r"^\s*(References|See also|External links|Notes|Bibliography|Further reading)",
    re.IGNORECASE
)


def dot_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute similarity between two vectors using dot product.

    For normalized vectors (L2 norm = 1), dot product equals cosine similarity.
    This is faster than computing full cosine similarity.

    Args:
        vec_a: First vector (should be normalized).
        vec_b: Second vector (should be normalized).

    Returns:
        Similarity score (dot product).
    """
    return float(np.dot(vec_a, vec_b))


class DynamicSemanticExpander(BaseNodePostprocessor):
    """
    Expand retrieved nodes by including semantically similar neighbors.

    This post-processor takes retrieved sentence nodes and greedily expands
    the context by including neighboring sentences while their cosine
    similarity remains above a threshold.

    Uses a HYBRID WINDOW approach:
    - Always includes ±min_window neighbors (safety net)
    - Expands further only if similarity > threshold

    Performance optimizations:
    - Batch-fetches all potential neighbor nodes upfront (fixes N+1 I/O)
    - Pre-converts embeddings to numpy arrays once (fixes NumPy overhead)
    - Reuses node cache in deduplication (fixes Double Read)

    Attributes:
        docstore: Access to all indexed nodes for neighbor lookup.
        threshold: Minimum cosine similarity for expansion beyond min_window (default: 0.6).
        max_expand: Maximum nodes to include per direction (default: 5).
        min_window: Minimum neighbors to always include (default: 1).
        min_chunk_length: Filter out chunks shorter than this (default: 50).
    """

    docstore: BaseDocumentStore
    threshold: float = DEFAULT_EXPANSION_CONFIG.threshold
    skip_threshold: float = DEFAULT_EXPANSION_CONFIG.skip_threshold
    max_expand: int = DEFAULT_EXPANSION_CONFIG.max_expand
    min_window: int = DEFAULT_EXPANSION_CONFIG.min_window
    min_chunk_length: int = DEFAULT_EXPANSION_CONFIG.min_chunk_length
    relevance_threshold_pct: float = DEFAULT_EXPANSION_CONFIG.relevance_threshold_pct
    merge_gap: int = DEFAULT_EXPANSION_CONFIG.merge_gap
    target_clusters: int = DEFAULT_EXPANSION_CONFIG.target_clusters

    # Adaptive threshold parameters
    adaptive_threshold_enabled: bool = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.enabled
    decay_lambda_sparse: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.decay_lambda_sparse
    decay_lambda_dense: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.decay_lambda_dense
    density_threshold: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.density_threshold
    gradient_cliff_factor: float = DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.gradient_cliff_factor

    # Seed validation parameters
    seed_validation_enabled: bool = DEFAULT_SEED_VALIDATION_CONFIG.enabled
    seed_mod_z_threshold: float = DEFAULT_SEED_VALIDATION_CONFIG.mod_z_threshold
    seed_support_threshold_pct: float = DEFAULT_SEED_VALIDATION_CONFIG.support_threshold_pct
    seed_min_local_support: int = DEFAULT_SEED_VALIDATION_CONFIG.min_local_support
    seed_window_size: int = DEFAULT_SEED_VALIDATION_CONFIG.window_size
    seed_early_position_threshold: int = DEFAULT_SEED_VALIDATION_CONFIG.early_position_threshold
    seed_rejection_log_path: Optional[str] = None

    # Runtime caches (populated per request)
    _node_cache: dict = {}
    _embedding_cache: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def _prefetch_neighbors(self, seed_nodes: list[TextNode]) -> None:
        """
        Batch-fetch all potential neighbor nodes and cache embeddings.

        This solves the N+1 I/O problem by fetching all nodes we might need
        in a single batch, and converts embeddings to numpy arrays once.

        Args:
            seed_nodes: List of seed nodes to expand from.
        """
        # Clear caches for new request
        self._node_cache = {}
        self._embedding_cache = {}

        # Collect all node IDs we might need
        ids_to_fetch = set()

        for node in seed_nodes:
            # Add seed node
            ids_to_fetch.add(node.node_id)

            # Walk the linked list to collect potential neighbor IDs
            # We need max_expand + 1 in each direction (for skip logic)
            expand_range = self.max_expand + 1

            # Collect prev IDs
            current_id = node.node_id
            for _ in range(expand_range):
                # Extract position from ID (format: doc_sent_0042)
                try:
                    pos = int(current_id.split("_")[-1])
                    prev_pos = pos - 1
                    if prev_pos >= 0:
                        prev_id = f"doc_sent_{prev_pos:04d}"
                        ids_to_fetch.add(prev_id)
                        current_id = prev_id
                    else:
                        break
                except (ValueError, IndexError):
                    break

            # Collect next IDs
            current_id = node.node_id
            for _ in range(expand_range):
                try:
                    pos = int(current_id.split("_")[-1])
                    next_id = f"doc_sent_{pos + 1:04d}"
                    ids_to_fetch.add(next_id)
                    current_id = next_id
                except (ValueError, IndexError):
                    break

        # Batch-fetch all nodes at once
        for node_id in ids_to_fetch:
            try:
                node = self.docstore.get_node(node_id)
                if node:
                    self._node_cache[node_id] = node
                    # Pre-convert embedding to numpy array and normalize
                    if node.embedding is not None:
                        emb = np.array(node.embedding, dtype=np.float32)
                        # Normalize for dot-product similarity
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        self._embedding_cache[node_id] = emb
            except Exception:
                pass

    def _get_cached_node(self, node_id: str) -> Optional[TextNode]:
        """Get node from cache (O(1) lookup instead of I/O)."""
        return self._node_cache.get(node_id)

    def _get_cached_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get pre-converted numpy embedding from cache."""
        return self._embedding_cache.get(node_id)

    def _validate_seed(
        self,
        seed_idx: int,
        scores: np.ndarray,
        seed_node_id: str,
    ) -> bool:
        """
        Validate seed using Modified Z-score and local semantic support.

        Uses MAD (Median Absolute Deviation) for robust outlier detection,
        combined with "lone ranger" check to filter isolated high scores.

        Args:
            seed_idx: Index of the seed in the scores array.
            scores: Array of all seed scores.
            seed_node_id: Node ID of the seed for neighbor lookup.

        Returns:
            True if seed is valid, False otherwise.
        """
        if not self.seed_validation_enabled:
            return True, {}

        # Need at least N scores for meaningful statistics
        if len(scores) < DEFAULT_SEED_VALIDATION_CONFIG.min_scores_for_validation:
            return True, {}

        # 1. Statistical validation via Modified Z-score
        median_score = np.median(scores)
        mad = np.median(np.abs(scores - median_score))

        # Handle edge case where MAD is 0 (all scores identical)
        if mad < DEFAULT_SEED_VALIDATION_CONFIG.mad_epsilon:
            return True, {}

        mod_z = DEFAULT_SEED_VALIDATION_CONFIG.mad_constant * (scores[seed_idx] - median_score) / mad

        # If score is NOT a positive outlier, it's valid (we want high scores)
        # Modified Z > threshold means it's an outlier (unusually high)
        # But we also need to check if it has semantic support
        if mod_z <= self.seed_mod_z_threshold:
            # Not an outlier, automatically valid
            return True, {"mod_z": mod_z, "is_outlier": False}

        # 2. Extract position from node ID
        try:
            seed_pos = int(seed_node_id.split("_")[-1])
        except (ValueError, IndexError):
            return True, {}  # Can't validate, assume valid

        # 3. Position-aware relaxation: first sentences are often important intros
        if seed_pos < self.seed_early_position_threshold:
            # Early sentences get a pass - they're usually topic introductions
            return True, {"mod_z": mod_z, "is_outlier": True, "position_exempt": True}

        # 4. Local semantic support check ("lone ranger" detection)
        support_count = 0
        support_threshold = scores[seed_idx] * self.seed_support_threshold_pct

        for offset in range(-self.seed_window_size, self.seed_window_size + 1):
            if offset == 0:
                continue  # Skip seed itself

            neighbor_pos = seed_pos + offset
            if neighbor_pos < 0:
                continue

            neighbor_id = f"doc_sent_{neighbor_pos:04d}"
            neighbor_emb = self._get_cached_embedding(neighbor_id)

            if neighbor_emb is None:
                continue

            # Get seed embedding for comparison
            seed_emb = self._get_cached_embedding(seed_node_id)
            if seed_emb is None:
                continue

            # Compute similarity between seed and neighbor
            neighbor_sim = dot_similarity(seed_emb, neighbor_emb)

            if neighbor_sim >= support_threshold:
                support_count += 1

        # Seed is valid if it has sufficient local support
        return support_count >= self.seed_min_local_support, {
            "mod_z": mod_z,
            "support_count": support_count,
            "seed_pos": seed_pos,
            "is_outlier": mod_z > self.seed_mod_z_threshold,
        }

    def _compute_adaptive_threshold(
        self,
        seed_score: float,
        local_scores: list[float],
        distance: int,
    ) -> float:
        """
        Compute adaptive expansion threshold based on distance and local density.

        Uses exponential decay with rate adjusted by local score density.
        In dense regions (many high scores), decay is slower to capture more context.

        Args:
            seed_score: The seed's similarity score.
            local_scores: Scores of already-collected neighbors.
            distance: Current distance from seed.

        Returns:
            Adaptive threshold for this distance.
        """
        if not self.adaptive_threshold_enabled or not local_scores:
            return self.threshold

        # Estimate local density: ratio of high-scoring neighbors
        high_score_count = sum(
            1 for s in local_scores 
            if s > seed_score * DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.density_score_ratio
        )
        density = high_score_count / len(local_scores) if local_scores else 0

        # Select decay rate based on density
        decay_lambda = self.decay_lambda_dense if density > self.density_threshold else self.decay_lambda_sparse

        # Compute decayed threshold
        decay_threshold = seed_score * np.exp(-decay_lambda * distance)

        # Floor: don't go below base threshold
        return max(decay_threshold, self.threshold * DEFAULT_ADAPTIVE_THRESHOLD_CONFIG.floor_multiplier)

    def _should_stop_gradient(self, scores_sequence: list[float]) -> bool:
        """
        Detect semantic cliff via gradient acceleration.

        Stops expansion when the similarity drop accelerates significantly,
        indicating we've crossed a topic boundary.

        Args:
            scores_sequence: List of similarity scores during expansion.

        Returns:
            True if expansion should stop due to cliff detection.
        """
        if not self.adaptive_threshold_enabled:
            return False

        if len(scores_sequence) < 3:
            return False

        # Compute gradients (differences between consecutive scores)
        gradients = np.diff(scores_sequence)

        if len(gradients) < 2:
            return False

        # Avoid division by zero
        prev_gradient = gradients[-2] if abs(gradients[-2]) > 1e-10 else 1e-10

        # Gradient change ratio
        gradient_change = gradients[-1] / prev_gradient

        # If drop accelerated beyond threshold — stop
        return gradient_change < self.gradient_cliff_factor

    def _log_rejected_seed(
        self,
        query: str,
        node: TextNode,
        score: float,
        rejection_info: dict,
    ) -> None:
        """
        Log rejected seed to file for analysis.

        Args:
            query: The query string.
            node: The rejected seed node.
            score: The seed's retrieval score.
            rejection_info: Dict with mod_z, support_count, etc.
        """
        if not self.seed_rejection_log_path:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "seed_id": node.node_id,
            "seed_text": node.text[:500] if node.text else "",  # Truncate for readability
            "score": float(score) if score else 0.0,
            "mod_z": float(rejection_info.get("mod_z")) if rejection_info.get("mod_z") is not None else None,
            "support_count": int(rejection_info.get("support_count")) if rejection_info.get("support_count") is not None else None,
            "is_outlier": bool(rejection_info.get("is_outlier")) if rejection_info.get("is_outlier") is not None else None,
        }

        log_path = Path(self.seed_rejection_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to JSONL file (one JSON object per line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """
        Process retrieved nodes and expand context based on semantic similarity.

        Uses Query-Aware Expansion: includes neighbors that are directly relevant
        to the query, even if adjacency similarity is below threshold.
        """
        if not nodes:
            return []

        # Prefetch all potential neighbors in one batch (fixes N+1 I/O)
        seed_nodes = [ns.node for ns in nodes]
        self._prefetch_neighbors(seed_nodes)

        # Compute query embedding for Query-Aware Expansion
        query_embedding = None
        relevance_threshold = self.threshold  # Fallback to adjacency threshold

        if query_bundle and query_bundle.query_str:
            from llama_index.core import Settings
            query_emb = Settings.embed_model.get_text_embedding(query_bundle.query_str)
            query_embedding = np.array(query_emb, dtype=np.float32)
            # Normalize for dot-product similarity
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            # Compute relevance_threshold as % of max seed relevance score
            max_score = max(ns.score for ns in nodes if ns.score is not None)
            if max_score and max_score > 0:
                relevance_threshold = max_score * self.relevance_threshold_pct

        # Validate seeds before expansion
        scores_array = np.array([ns.score for ns in nodes if ns.score is not None])
        valid_nodes = []
        query_str = query_bundle.query_str if query_bundle else ""

        for idx, node_score in enumerate(nodes):
            is_valid, rejection_info = self._validate_seed(
                seed_idx=idx,
                scores=scores_array,
                seed_node_id=node_score.node.node_id,
            )
            if is_valid:
                valid_nodes.append(node_score)
            else:
                # Log rejected seed
                self._log_rejected_seed(
                    query=query_str,
                    node=node_score.node,
                    score=node_score.score or 0.0,
                    rejection_info=rejection_info,
                )

        expanded = []

        for node_score in valid_nodes:
            cluster = self._expand_cluster(
                node_score.node,
                query_embedding=query_embedding,
                relevance_threshold=relevance_threshold,
            )

            # Filter garbage chunks
            cluster = self._filter_garbage(cluster)

            if not cluster:
                continue

            merged_text = self._merge_cluster(cluster)

            new_node = TextNode(
                text=merged_text,
                metadata={
                    "source_nodes": [n.node_id for n in cluster],
                    "expansion_size": len(cluster),
                },
            )
            expanded.append(NodeWithScore(node=new_node, score=node_score.score))

        return self._deduplicate(
            expanded,
            query_embedding=query_embedding,
            relevance_threshold=relevance_threshold,
            target_k=self.target_clusters,
        )

    def _expand_cluster(
        self,
        seed_node: TextNode,
        query_embedding: Optional[np.ndarray] = None,
        relevance_threshold: float = 0.6,
    ) -> list[TextNode]:
        """
        Expand context around seed node using HYBRID WINDOW approach with Adaptive Threshold.

        Uses Query-Aware Expansion: if a neighbor fails adjacency check but is
        directly relevant to the query (similarity >= relevance_threshold),
        include it anyway.

        Uses Adaptive Threshold: threshold decays with distance, adjusted by local density.
        Uses Gradient Stop: stops on semantic cliff (sharp acceleration in score drop).

        Args:
            seed_node: The starting node to expand from.
            query_embedding: Optional query embedding for Query-Aware Expansion.
            relevance_threshold: Threshold for query relevance (% of max score).

        Returns:
            List of nodes in the expanded cluster.
        """
        cluster = [seed_node]

        # Get cached embedding for seed
        seed_id = seed_node.node_id
        seed_emb = self._get_cached_embedding(seed_id)

        # Get seed's similarity score for adaptive threshold
        seed_score = 1.0  # Default if we can't compute
        if query_embedding is not None and seed_emb is not None:
            seed_score = dot_similarity(query_embedding, seed_emb)

        # Track scores for gradient detection
        left_scores = []
        right_scores = []

        # Expand left
        current_id = seed_id
        current_emb = seed_emb

        for i in range(self.max_expand):
            prev_id = seed_node.metadata.get("prev_id")
            if not prev_id:
                # Try to derive from position
                try:
                    pos = int(current_id.split("_")[-1])
                    if pos > 0:
                        prev_id = f"doc_sent_{pos - 1:04d}"
                    else:
                        break
                except (ValueError, IndexError):
                    break

            prev_node = self._get_cached_node(prev_id)
            if prev_node is None:
                break

            prev_emb = self._get_cached_embedding(prev_id)

            # HYBRID: Always take min_window neighbors
            if i >= self.min_window:
                if prev_emb is None or current_emb is None:
                    break

                # Use dot product (vectors are pre-normalized)
                similarity = dot_similarity(current_emb, prev_emb)
                left_scores.append(similarity)

                # Check gradient stop
                if self._should_stop_gradient(left_scores):
                    break

                # Compute adaptive threshold
                adaptive_thresh = self._compute_adaptive_threshold(
                    seed_score, left_scores, i + 1
                )

                if similarity < adaptive_thresh:
                    # Query-Aware Expansion: check if neighbor is query-relevant
                    query_relevant = False
                    if query_embedding is not None and prev_emb is not None:
                        query_sim = dot_similarity(query_embedding, prev_emb)
                        if query_sim >= relevance_threshold:
                            query_relevant = True

                    if not query_relevant:
                        # SENTENCE SKIP LOGIC
                        # Try to bridge if the NEXT neighbor is very strong
                        try:
                            pprev_pos = int(prev_id.split("_")[-1]) - 1
                            if pprev_pos >= 0:
                                pprev_id = f"doc_sent_{pprev_pos:04d}"
                                pprev_node = self._get_cached_node(pprev_id)
                                pprev_emb = self._get_cached_embedding(pprev_id)
                                if pprev_node and pprev_emb is not None:
                                    p_similarity = dot_similarity(current_emb, pprev_emb)
                                    if p_similarity >= self.skip_threshold:
                                        # Bridge gap: add both and continue
                                        cluster.insert(0, prev_node)
                                        cluster.insert(0, pprev_node)
                                        current_id = pprev_id
                                        current_emb = pprev_emb
                                        left_scores.append(p_similarity)
                                        continue
                        except (ValueError, IndexError):
                            pass
                        break

            cluster.insert(0, prev_node)
            current_id = prev_id
            current_emb = prev_emb

        # Expand right
        current_id = seed_id
        current_emb = seed_emb

        for i in range(self.max_expand):
            # Derive next_id from position
            try:
                pos = int(current_id.split("_")[-1])
                next_id = f"doc_sent_{pos + 1:04d}"
            except (ValueError, IndexError):
                break

            next_node = self._get_cached_node(next_id)
            if next_node is None:
                break

            next_emb = self._get_cached_embedding(next_id)

            # HYBRID: Always take min_window neighbors
            if i >= self.min_window:
                if next_emb is None or current_emb is None:
                    break

                # Use dot product (vectors are pre-normalized)
                similarity = dot_similarity(current_emb, next_emb)
                right_scores.append(similarity)

                # Check gradient stop
                if self._should_stop_gradient(right_scores):
                    break

                # Compute adaptive threshold
                adaptive_thresh = self._compute_adaptive_threshold(
                    seed_score, right_scores, i + 1
                )

                if similarity < adaptive_thresh:
                    # Query-Aware Expansion: check if neighbor is query-relevant
                    query_relevant = False
                    if query_embedding is not None and next_emb is not None:
                        query_sim = dot_similarity(query_embedding, next_emb)
                        if query_sim >= relevance_threshold:
                            query_relevant = True

                    if not query_relevant:
                        # SENTENCE SKIP LOGIC
                        # Try to bridge if the NEXT neighbor is very strong
                        try:
                            nnext_pos = int(next_id.split("_")[-1]) + 1
                            nnext_id = f"doc_sent_{nnext_pos:04d}"
                            nnext_node = self._get_cached_node(nnext_id)
                            nnext_emb = self._get_cached_embedding(nnext_id)
                            if nnext_node and nnext_emb is not None:
                                n_similarity = dot_similarity(current_emb, nnext_emb)
                                if n_similarity >= self.skip_threshold:
                                    # Bridge gap: add both and continue
                                    cluster.append(next_node)
                                    cluster.append(nnext_node)
                                    current_id = nnext_id
                                    current_emb = nnext_emb
                                    right_scores.append(n_similarity)
                                    continue
                        except (ValueError, IndexError):
                            pass
                        break

            cluster.append(next_node)
            current_id = next_id
            current_emb = next_emb

        return cluster

    def _filter_garbage(self, cluster: list[TextNode]) -> list[TextNode]:
        """
        Filter out garbage chunks (too short or reference sections).

        Args:
            cluster: List of nodes to filter.

        Returns:
            Filtered list of nodes.
        """
        result = []
        for node in cluster:
            text = node.text.strip()

            # Skip too short chunks
            if len(text) < self.min_chunk_length:
                continue

            # Skip reference/links sections
            if GARBAGE_PATTERNS.match(text):
                continue

            result.append(node)

        return result

    def _merge_cluster(self, cluster: list[TextNode]) -> str:
        """Merge cluster nodes into a single text string."""
        return " ".join(node.text for node in cluster)

    def _deduplicate(
        self,
        nodes: list[NodeWithScore],
        query_embedding: Optional[np.ndarray] = None,
        relevance_threshold: float = 0.6,
        target_k: int = 5,
    ) -> list[NodeWithScore]:
        """
        Merge overlapping clusters into unified context blocks.

        If two clusters share source nodes, they are merged into one larger cluster.
        Uses node_cache instead of re-fetching from docstore (fixes Double Read).

        If merge reduces cluster count below target_k, backfills with next-closest
        sentences to the query that aren't in any existing cluster.

        Args:
            nodes: List of expanded nodes.
            query_embedding: Query embedding for backfill ranking.
            relevance_threshold: Threshold for query relevance.
            target_k: Target number of clusters to return.

        Returns:
            List of merged, non-overlapping nodes.
        """
        if not nodes:
            return []

        # Build clusters with their source IDs and positions
        clusters = []
        for node_score in nodes:
            source_ids = node_score.node.metadata.get("source_nodes", [])
            # Extract positions from node IDs (format: doc_sent_0042)
            positions = []
            for sid in source_ids:
                try:
                    pos = int(sid.split("_")[-1])
                    positions.append(pos)
                except (ValueError, IndexError):
                    pass

            clusters.append({
                "node_score": node_score,
                "source_ids": set(source_ids),
                "min_pos": min(positions) if positions else 0,
                "max_pos": max(positions) if positions else 0,
                "score": node_score.score or 0.0,
            })

        # Merge overlapping clusters using Union-Find approach
        merged = []
        used = [False] * len(clusters)

        for i, cluster_i in enumerate(clusters):
            if used[i]:
                continue

            # Start a new merged group
            merged_ids = set(cluster_i["source_ids"])
            merged_min = cluster_i["min_pos"]
            merged_max = cluster_i["max_pos"]
            best_score = cluster_i["score"]
            used[i] = True

            # Keep merging until no more overlaps found
            changed = True
            while changed:
                changed = False
                for j, cluster_j in enumerate(clusters):
                    if used[j]:
                        continue
                    # Check if overlaps or within merge_gap (Multi-Seed Merging)
                    if (cluster_j["source_ids"] & merged_ids or
                        cluster_j["min_pos"] <= merged_max + self.merge_gap and cluster_j["max_pos"] >= merged_min - self.merge_gap):
                        merged_ids.update(cluster_j["source_ids"])
                        merged_min = min(merged_min, cluster_j["min_pos"])
                        merged_max = max(merged_max, cluster_j["max_pos"])
                        best_score = max(best_score, cluster_j["score"])
                        used[j] = True
                        changed = True

            # Reconstruct merged text from CACHE (not docstore!) - fixes Double Read
            merged_nodes = []
            all_node_ids = set()
            for pos in range(merged_min, merged_max + 1):
                node_id = f"doc_sent_{pos:04d}"
                # Use cache instead of docstore.get_node()
                node = self._get_cached_node(node_id)
                if node:
                    merged_nodes.append(node)
                    all_node_ids.add(node_id)

            if merged_nodes:
                merged_text = " ".join(n.text for n in merged_nodes)
                new_node = TextNode(
                    text=merged_text,
                    metadata={
                        "source_nodes": list(all_node_ids),
                        "expansion_size": len(all_node_ids),
                        "merged_from": len([c for c in clusters if c["source_ids"] & merged_ids]),
                        "gap_filled": len(all_node_ids) - len(merged_ids),
                    },
                )
                merged.append(NodeWithScore(node=new_node, score=best_score))

        # BACKFILL: If merge reduced count below target_k, find additional seeds
        if len(merged) < target_k and query_embedding is not None:
            # Collect all positions already in merged clusters
            used_positions = set()
            for node_score in merged:
                source_ids = node_score.node.metadata.get("source_nodes", [])
                for sid in source_ids:
                    try:
                        pos = int(sid.split("_")[-1])
                        used_positions.add(pos)
                    except (ValueError, IndexError):
                        pass

            # Find candidates from cache that aren't in any cluster
            candidates = []
            for node_id, emb in self._embedding_cache.items():
                try:
                    pos = int(node_id.split("_")[-1])
                    if pos not in used_positions:
                        # Compute similarity to query
                        sim = dot_similarity(query_embedding, emb)
                        candidates.append((node_id, pos, sim))
                except (ValueError, IndexError):
                    pass

            # Sort by query similarity (descending)
            candidates.sort(key=lambda x: x[2], reverse=True)

            # Expand top candidates until we reach target_k
            for node_id, pos, sim in candidates:
                if len(merged) >= target_k:
                    break

                # Skip if this position is now used (from previous backfill)
                if pos in used_positions:
                    continue

                seed_node = self._get_cached_node(node_id)
                if not seed_node:
                    continue

                # Expand this seed into a cluster
                cluster = self._expand_cluster(
                    seed_node,
                    query_embedding=query_embedding,
                    relevance_threshold=relevance_threshold,
                )
                cluster = self._filter_garbage(cluster)

                if not cluster:
                    continue

                # Mark positions as used
                cluster_ids = set()
                for n in cluster:
                    cluster_ids.add(n.node_id)
                    try:
                        p = int(n.node_id.split("_")[-1])
                        used_positions.add(p)
                    except (ValueError, IndexError):
                        pass

                # Create merged node
                merged_text = self._merge_cluster(cluster)
                new_node = TextNode(
                    text=merged_text,
                    metadata={
                        "source_nodes": list(cluster_ids),
                        "expansion_size": len(cluster_ids),
                        "backfill": True,
                    },
                )
                merged.append(NodeWithScore(node=new_node, score=sim))

        return merged
