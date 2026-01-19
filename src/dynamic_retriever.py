"""Dynamic Semantic Expander - Custom NodePostprocessor for context expansion."""

from typing import Optional

import numpy as np
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore import BaseDocumentStore


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

    Attributes:
        docstore: Access to all indexed nodes for neighbor lookup.
        threshold: Minimum cosine similarity for expansion (default: 0.75).
        max_expand: Maximum nodes to include per direction (default: 5).
    """

    docstore: BaseDocumentStore
    threshold: float = 0.75
    max_expand: int = 5

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
        Expand context around seed node by including similar neighbors.

        Args:
            seed_node: The starting node to expand from.

        Returns:
            List of nodes forming the expanded cluster.
        """
        cluster = [seed_node]

        # Expand left
        current = seed_node
        for _ in range(self.max_expand):
            prev_id = current.metadata.get("prev_id")
            if not prev_id:
                break
            prev_node = self.docstore.get_node(prev_id)
            if prev_node is None or prev_node.embedding is None:
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
        for _ in range(self.max_expand):
            next_id = current.metadata.get("next_id")
            if not next_id:
                break
            next_node = self.docstore.get_node(next_id)
            if next_node is None or next_node.embedding is None:
                break

            similarity = cosine_similarity(
                np.array(current.embedding), np.array(next_node.embedding)
            )
            if similarity < self.threshold:
                break

            cluster.append(next_node)
            current = next_node

        return cluster

    def _merge_cluster(self, cluster: list[TextNode]) -> str:
        """Merge cluster nodes into a single text string."""
        return " ".join(node.text for node in cluster)

    def _deduplicate(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Remove duplicate clusters that share common source nodes.

        Args:
            nodes: List of expanded nodes.

        Returns:
            Deduplicated list of nodes.
        """
        seen_ids: set[str] = set()
        result = []

        for node_score in nodes:
            source_ids = set(node_score.node.metadata.get("source_nodes", []))
            if not source_ids & seen_ids:
                result.append(node_score)
                seen_ids.update(source_ids)

        return result
