"""Cached Dynamic Expander for Optuna optimization.

This module provides a lightweight implementation of the Dynamic Semantic Expander
that uses only pre-computed similarity matrices. Zero model.encode() calls - 
all computation is done via numpy array lookups.
"""

from dataclasses import dataclass

import numpy as np

from src.config import DEFAULT_EXPANSION_CONFIG


@dataclass
class ExpandedCluster:
    """Result of cluster expansion.
    
    Attributes:
        start_idx: Starting sentence index (inclusive).
        end_idx: Ending sentence index (inclusive).
        seed_idx: Index of the seed sentence.
        score: Seed's similarity score to query.
    """
    start_idx: int
    end_idx: int
    seed_idx: int
    score: float


class CachedDynamicExpander:
    """
    Dynamic Semantic Expander using pre-computed similarities only.
    
    This class is designed for fast Optuna trials. All similarity lookups
    are O(1) numpy array accesses. No embedding model calls are made.
    
    The expansion algorithm is equivalent to DynamicSemanticExpander but uses:
    - neighbor_sims[i] instead of cosine(embed[i], embed[i+1])
    - sentence_sims[i] instead of cosine(query_embed, sent_embed[i])
    
    Args:
        neighbor_sims: Pre-computed adjacent sentence similarities, shape (n-1,).
        sentence_sims: Pre-computed query-sentence similarities, shape (n,).
        top_k_indices: Pre-computed top-K candidate indices.
        threshold: Minimum similarity for expansion beyond min_window.
        skip_threshold: Threshold for sentence skipping (bridging gaps).
        min_window: Minimum neighbors to always include.
        max_expand: Maximum expansion in each direction.
        relevance_threshold_pct: Query relevance threshold as % of max seed score.
        merge_gap: Maximum gap between clusters to merge.
        target_clusters: Target number of output clusters.
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
    
    def expand_and_retrieve(self) -> list[ExpandedCluster]:
        """
        Expand clusters from top-K candidates with iterative backfill.
        
        Guarantees target_clusters output (if enough candidates exist) by:
        1. Process initial batch of candidates
        2. Expand + merge
        3. If < target_clusters, find uncovered candidates and repeat
        
        Returns:
            List of ExpandedCluster objects representing context windows.
        """
        if len(self.top_k_indices) == 0:
            return []
        
        # Compute relevance threshold from best candidate
        max_seed_score = self.sentence_sims[self.top_k_indices[0]]
        relevance_threshold = max_seed_score * self.relevance_threshold_pct
        
        # Track which candidates have been processed
        processed_candidates = set()
        # Track indices covered by existing clusters
        covered_indices = set()
        
        final_clusters: list[ExpandedCluster] = []
        candidate_ptr = 0  # Pointer into top_k_indices
        
        # Iteratively add candidates until we reach target_clusters
        while len(final_clusters) < self.target_clusters and candidate_ptr < len(self.top_k_indices):
            # Collect batch of new candidates not yet covered
            new_seeds = []
            while len(new_seeds) < self.target_clusters and candidate_ptr < len(self.top_k_indices):
                candidate_idx = self.top_k_indices[candidate_ptr]
                candidate_ptr += 1
                
                # Skip if already processed or covered by existing cluster
                if candidate_idx in processed_candidates:
                    continue
                if candidate_idx in covered_indices:
                    continue
                
                processed_candidates.add(candidate_idx)
                new_seeds.append(candidate_idx)
            
            if not new_seeds:
                break  # No more valid candidates
            
            # Expand new seeds
            new_clusters: list[ExpandedCluster] = []
            for seed_idx in new_seeds:
                cluster = self._expand_cluster(seed_idx, relevance_threshold)
                if cluster:
                    new_clusters.append(cluster)
            
            if not new_clusters:
                continue  # All expansions failed, try more candidates
            
            # Merge new clusters with existing ones
            all_clusters = final_clusters + new_clusters
            merged = self._merge_clusters(all_clusters)
            
            # Update final clusters and covered indices
            final_clusters = merged
            covered_indices = self.get_all_retrieved_indices(final_clusters)
        
        # Return up to target_clusters (sorted by score from merge)
        return final_clusters[:self.target_clusters]
    
    def _expand_cluster(
        self,
        seed_idx: int,
        relevance_threshold: float,
    ) -> ExpandedCluster | None:
        """
        Expand context around a single seed using hybrid window approach.
        
        Args:
            seed_idx: Index of the seed sentence.
            relevance_threshold: Minimum query relevance for inclusion.
        
        Returns:
            ExpandedCluster or None if seed is invalid.
        """
        if seed_idx < 0 or seed_idx >= self.num_sentences:
            return None
        
        seed_score = self.sentence_sims[seed_idx]
        
        # Expand left
        left_idx = seed_idx
        for i in range(self.max_expand):
            next_left = left_idx - 1
            if next_left < 0:
                break
            
            # HYBRID: Always include min_window neighbors
            if i >= self.min_window:
                # Check adjacency similarity (neighbor_sims[next_left] = sim(next_left, next_left+1))
                adj_sim = self.neighbor_sims[next_left] if next_left < len(self.neighbor_sims) else 0.0
                
                if adj_sim < self.threshold:
                    # Check query relevance as fallback
                    query_sim = self.sentence_sims[next_left]
                    if query_sim < relevance_threshold:
                        # Try skip logic
                        if next_left - 1 >= 0:
                            skip_sim = self.neighbor_sims[next_left - 1] if next_left - 1 < len(self.neighbor_sims) else 0.0
                            if skip_sim >= self.skip_threshold:
                                left_idx = next_left - 1  # Skip one sentence
                                continue
                        break
            
            left_idx = next_left
        
        # Expand right
        right_idx = seed_idx
        for i in range(self.max_expand):
            next_right = right_idx + 1
            if next_right >= self.num_sentences:
                break
            
            # HYBRID: Always include min_window neighbors
            if i >= self.min_window:
                # Check adjacency similarity (neighbor_sims[right_idx] = sim(right_idx, right_idx+1))
                adj_sim = self.neighbor_sims[right_idx] if right_idx < len(self.neighbor_sims) else 0.0
                
                if adj_sim < self.threshold:
                    # Check query relevance as fallback
                    query_sim = self.sentence_sims[next_right]
                    if query_sim < relevance_threshold:
                        # Try skip logic
                        if next_right + 1 < self.num_sentences and right_idx + 1 < len(self.neighbor_sims):
                            skip_sim = self.neighbor_sims[right_idx + 1]
                            if skip_sim >= self.skip_threshold:
                                right_idx = next_right + 1  # Skip one sentence
                                continue
                        break
            
            right_idx = next_right
        
        return ExpandedCluster(
            start_idx=left_idx,
            end_idx=right_idx,
            seed_idx=seed_idx,
            score=seed_score,
        )
    
    def _merge_clusters(self, clusters: list[ExpandedCluster]) -> list[ExpandedCluster]:
        """
        Merge overlapping or adjacent clusters.
        
        Uses Union-Find approach to merge clusters that:
        - Overlap (any shared indices)
        - Are within merge_gap of each other
        
        Args:
            clusters: List of expanded clusters.
        
        Returns:
            Merged list of clusters, sorted by score descending.
        """
        if not clusters:
            return []
        
        # Sort by seed score descending
        clusters = sorted(clusters, key=lambda c: c.score, reverse=True)
        
        merged: list[ExpandedCluster] = []
        used = [False] * len(clusters)
        
        for i, cluster_i in enumerate(clusters):
            if used[i]:
                continue
            
            # Start a new merged group
            merged_start = cluster_i.start_idx
            merged_end = cluster_i.end_idx
            best_score = cluster_i.score
            best_seed = cluster_i.seed_idx
            used[i] = True
            
            # Keep merging until no more overlaps
            changed = True
            while changed:
                changed = False
                for j, cluster_j in enumerate(clusters):
                    if used[j]:
                        continue
                    
                    # Check overlap or within merge_gap
                    if (cluster_j.start_idx <= merged_end + self.merge_gap and
                            cluster_j.end_idx >= merged_start - self.merge_gap):
                        merged_start = min(merged_start, cluster_j.start_idx)
                        merged_end = max(merged_end, cluster_j.end_idx)
                        if cluster_j.score > best_score:
                            best_score = cluster_j.score
                            best_seed = cluster_j.seed_idx
                        used[j] = True
                        changed = True
            
            merged.append(ExpandedCluster(
                start_idx=merged_start,
                end_idx=merged_end,
                seed_idx=best_seed,
                score=best_score,
            ))
        
        # Sort by score descending
        return sorted(merged, key=lambda c: c.score, reverse=True)
    
    def get_cluster_sentence_indices(self, cluster: ExpandedCluster) -> list[int]:
        """Get list of sentence indices in a cluster."""
        return list(range(cluster.start_idx, cluster.end_idx + 1))
    
    def get_all_retrieved_indices(self, clusters: list[ExpandedCluster]) -> set[int]:
        """Get all unique sentence indices across all clusters."""
        indices = set()
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
        Dictionary with metrics: hit (bool), tokens (int), num_clusters (int).
    """
    if answer_sentence_idx < 0:
        return {"hit": False, "tokens": 0, "num_clusters": len(clusters), "rank": -1}
    
    # Check if answer is in any cluster
    hit = False
    rank = -1
    
    for i, cluster in enumerate(clusters):
        if cluster.start_idx <= answer_sentence_idx <= cluster.end_idx:
            hit = True
            rank = i + 1  # 1-indexed rank
            break
    
    # Count tokens (rough: chars / 4)
    total_chars = 0
    for cluster in clusters:
        for idx in range(cluster.start_idx, cluster.end_idx + 1):
            if 0 <= idx < len(sentences):
                total_chars += len(sentences[idx])
    tokens = total_chars // 4
    
    return {
        "hit": hit,
        "tokens": tokens,
        "num_clusters": len(clusters),
        "rank": rank,
    }
