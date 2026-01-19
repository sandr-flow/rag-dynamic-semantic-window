"""Dynamic Semantic Expander - Custom NodePostprocessor for context expansion."""

import re
from typing import Optional

import numpy as np
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore import BaseDocumentStore


# Patterns to filter garbage chunks (references, links, etc.)
GARBAGE_PATTERNS = re.compile(
    r"^\s*(References|See also|External links|Notes|Bibliography|Further reading)",
    re.IGNORECASE
)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class DynamicSemanticExpander(BaseNodePostprocessor):
    """
    Expand retrieved nodes by including semantically similar neighbors.

    This post-processor takes retrieved sentence nodes and greedily expands
    the context by including neighboring sentences while their cosine
    similarity remains above a threshold.

    Uses a HYBRID WINDOW approach:
    - Always includes ±min_window neighbors (safety net)
    - Expands further only if similarity > threshold

    Attributes:
        docstore: Access to all indexed nodes for neighbor lookup.
        threshold: Minimum cosine similarity for expansion beyond min_window (default: 0.6).
        max_expand: Maximum nodes to include per direction (default: 5).
        min_window: Minimum neighbors to always include (default: 1).
        min_chunk_length: Filter out chunks shorter than this (default: 50).
    """

    docstore: BaseDocumentStore
    threshold: float = 0.6  # Lowered from 0.75
    max_expand: int = 5
    min_window: int = 1  # Always take ±1 neighbor
    min_chunk_length: int = 20  # Filter short garbage

    class Config:
        arbitrary_types_allowed = True

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """
        Process retrieved nodes and expand context based on semantic similarity.

        Args:
            nodes: List of retrieved nodes with scores.
            query_bundle: Optional query bundle (unused in current implementation).

        Returns:
            List of expanded nodes with merged context.
        """
        expanded = []

        for node_score in nodes:
            cluster = self._expand_cluster(node_score.node)
            
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

        return self._deduplicate(expanded)

    def _expand_cluster(self, seed_node: TextNode) -> list[TextNode]:
        """
        Expand context around seed node using HYBRID WINDOW approach.

        - Always includes ±min_window neighbors (safety net)
        - Expands further only if similarity > threshold

        Args:
            seed_node: The starting node to expand from.

        Returns:
            List of nodes forming the expanded cluster.
        """
        cluster = [seed_node]

        # Expand left
        current = seed_node
        for i in range(self.max_expand):
            prev_id = current.metadata.get("prev_id")
            if not prev_id:
                break
            prev_node = self.docstore.get_node(prev_id)
            if prev_node is None:
                break

            # HYBRID: Always take min_window neighbors, then check similarity
            if i >= self.min_window:
                if prev_node.embedding is None or current.embedding is None:
                    break
                similarity = cosine_similarity(
                    np.array(current.embedding), np.array(prev_node.embedding)
                )
                if similarity < self.threshold:
                    break

            cluster.insert(0, prev_node)
            current = prev_node

        # Expand right
        current = seed_node
        for i in range(self.max_expand):
            next_id = current.metadata.get("next_id")
            if not next_id:
                break
            next_node = self.docstore.get_node(next_id)
            if next_node is None:
                break

            # HYBRID: Always take min_window neighbors, then check similarity
            if i >= self.min_window:
                if next_node.embedding is None or current.embedding is None:
                    break
                similarity = cosine_similarity(
                    np.array(current.embedding), np.array(next_node.embedding)
                )
                if similarity < self.threshold:
                    break

            cluster.append(next_node)
            current = next_node

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

    def _deduplicate(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Merge overlapping clusters into unified context blocks.

        If two clusters share source nodes, they are merged into one larger cluster.

        Args:
            nodes: List of expanded nodes.

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
                    # Check if overlaps or adjacent
                    if (cluster_j["source_ids"] & merged_ids or 
                        cluster_j["min_pos"] <= merged_max + 1 and cluster_j["max_pos"] >= merged_min - 1):
                        merged_ids.update(cluster_j["source_ids"])
                        merged_min = min(merged_min, cluster_j["min_pos"])
                        merged_max = max(merged_max, cluster_j["max_pos"])
                        best_score = max(best_score, cluster_j["score"])
                        used[j] = True
                        changed = True

            # Reconstruct merged text from docstore
            merged_nodes = []
            for pos in range(merged_min, merged_max + 1):
                node_id = f"doc_sent_{pos:04d}"
                if node_id in merged_ids:
                    try:
                        node = self.docstore.get_node(node_id)
                        if node:
                            merged_nodes.append(node)
                    except Exception:
                        pass

            if merged_nodes:
                merged_text = " ".join(n.text for n in merged_nodes)
                new_node = TextNode(
                    text=merged_text,
                    metadata={
                        "source_nodes": list(merged_ids),
                        "expansion_size": len(merged_ids),
                        "merged_from": len([c for c in clusters if c["source_ids"] & merged_ids]),
                    },
                )
                merged.append(NodeWithScore(node=new_node, score=best_score))

        return merged
