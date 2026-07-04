"""Tests for the unified Dynamic Semantic expansion core and its adapter.

Covers:
- the 0.1 regression (left expansion was capped at one sentence) as
  behavioral specs on the adapter;
- parity: on randomized corpora the LlamaIndex adapter and the raw-array
  core (as driven by the HPO objective) must produce identical clusters.
"""

import numpy as np
import pytest
from llama_index.core.schema import NodeWithScore

from src.dynamic_retriever import DynamicSemanticExpander
from src.expansion_core import DynamicExpansionCore, build_garbage_mask
from src.utils import create_sentence_nodes


def _make_expander(sentences: list[str], embeddings: np.ndarray, **params):
    nodes = create_sentence_nodes(sentences)
    expander = DynamicSemanticExpander(
        sentences=sentences,
        node_ids=[node.node_id for node in nodes],
        embeddings=embeddings,
        **params,
    )
    return expander, nodes


def _source_positions(node_score: NodeWithScore) -> list[int]:
    return [
        int(node_id.split("_")[-1])
        for node_id in node_score.node.metadata["source_nodes"]
    ]


# ---------------------------------------------------------------------------
# Behavioral specs (0.1 regression: left expansion must walk, not repeat)
# ---------------------------------------------------------------------------


def _uniform_corpus(num_sentences: int):
    sentences = [
        f"Sentence number {i} about one single shared topic." for i in range(num_sentences)
    ]
    embeddings = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (num_sentences, 1))
    return sentences, embeddings


def test_left_expansion_reaches_max_expand_without_duplicates():
    sentences, embeddings = _uniform_corpus(12)
    expander, nodes = _make_expander(sentences, embeddings, max_expand=3, min_window=1)

    result = expander._postprocess_nodes([NodeWithScore(node=nodes[6], score=0.9)], None)

    assert len(result) == 1
    positions = _source_positions(result[0])
    assert len(positions) == len(set(positions)), "cluster must not contain duplicates"
    # Uniform similarity (1.0) must let both directions reach max_expand.
    assert sorted(positions) == list(range(3, 10))
    assert result[0].node.metadata["expansion_size"] == 7


def test_expansion_stops_at_document_start():
    sentences, embeddings = _uniform_corpus(8)
    expander, nodes = _make_expander(sentences, embeddings, max_expand=3, min_window=1)

    result = expander._postprocess_nodes([NodeWithScore(node=nodes[0], score=0.9)], None)

    assert len(result) == 1
    assert sorted(_source_positions(result[0])) == list(range(0, 4))


def test_expansion_does_not_cross_document_segment_boundary():
    sentences, embeddings = _uniform_corpus(6)
    expander, nodes = _make_expander(
        sentences,
        embeddings,
        doc_ids=["doc_a", "doc_a", "doc_a", "doc_b", "doc_b", "doc_b"],
        max_expand=4,
        min_window=1,
    )

    result = expander._postprocess_nodes([NodeWithScore(node=nodes[2], score=0.9)], None)

    assert len(result) == 1
    assert _source_positions(result[0]) == [0, 1, 2]
    assert result[0].node.metadata["source_doc"] == "doc_a"


# ---------------------------------------------------------------------------
# Dual-space adjacency (P.3): neighbor sims from a separate clean matrix
# ---------------------------------------------------------------------------


def test_adjacency_embeddings_control_expansion_independently():
    """Uniform 'phantom' embeddings expand fully; orthogonal clean adjacency stops it."""
    sentences, phantom = _uniform_corpus(12)
    orthogonal = np.eye(12, 3, dtype=np.float32)  # every adjacent cosine is 0

    expander_default, nodes = _make_expander(
        sentences, phantom, max_expand=3, min_window=1, threshold=0.5
    )
    expander_dual = DynamicSemanticExpander(
        sentences=sentences,
        node_ids=[node.node_id for node in nodes],
        embeddings=phantom,
        adjacency_embeddings=orthogonal,
        max_expand=3,
        min_window=1,
        threshold=0.5,
    )

    seed = [NodeWithScore(node=nodes[6], score=0.9)]
    full = _source_positions(expander_default._postprocess_nodes(seed, None)[0])
    clipped = _source_positions(expander_dual._postprocess_nodes(seed, None)[0])

    assert sorted(full) == list(range(3, 10))  # phantom-only reaches max_expand
    assert sorted(clipped) == [5, 6, 7]  # clean adjacency stops at min_window


def test_adjacency_embeddings_shape_mismatch_rejected():
    sentences, phantom = _uniform_corpus(6)
    expander = DynamicSemanticExpander(
        sentences=sentences,
        node_ids=[f"node_{i}" for i in range(6)],
        embeddings=phantom,
        adjacency_embeddings=np.ones((5, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="adjacency_embeddings"):
        expander._ensure_prepared()


@pytest.mark.parametrize("seed", range(5))
def test_dual_space_adapter_matches_core(seed):
    """Parity holds when adjacency comes from a second (clean) matrix."""
    rng = np.random.default_rng(1000 + seed)
    sentences, phantom, query, params = _random_case(rng)
    clean = rng.normal(size=phantom.shape).astype(np.float32)
    clean /= np.linalg.norm(clean, axis=1, keepdims=True)

    neighbor_sims = np.sum(clean[:-1] * clean[1:], axis=1)
    sentence_sims = phantom @ query
    order = np.argsort(sentence_sims)[::-1].astype(np.int32)

    core = DynamicExpansionCore(
        neighbor_sims=neighbor_sims,
        sentence_sims=sentence_sims,
        top_k_indices=order,
        garbage_mask=build_garbage_mask(sentences),
        **params,
    )
    expected = core.expand_and_retrieve()

    nodes = create_sentence_nodes(sentences)
    expander = DynamicSemanticExpander(
        sentences=sentences,
        node_ids=[node.node_id for node in nodes],
        embeddings=phantom,
        adjacency_embeddings=clean,
        query_embedding=query,
        **params,
    )
    seeds = [
        NodeWithScore(node=nodes[int(pos)], score=float(sentence_sims[pos]))
        for pos in order[: min(10, len(sentences))]
    ]
    result = expander._postprocess_nodes(seeds, None)

    assert len(result) == len(expected)
    for node_score, cluster in zip(result, expected, strict=True):
        assert _source_positions(node_score) == list(
            range(cluster.start_idx, cluster.end_idx + 1)
        )


# ---------------------------------------------------------------------------
# Parity: adapter (LlamaIndex path) == core (HPO array path)
# ---------------------------------------------------------------------------


def _random_case(rng: np.random.Generator):
    num_sentences = int(rng.integers(30, 80))
    dim = 16

    embeddings = rng.normal(size=(num_sentences, dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    sentences = []
    for i in range(num_sentences):
        roll = rng.random()
        if roll < 0.05:
            sentences.append("Ok.")  # short -> garbage
        elif roll < 0.08:
            sentences.append("References and other external reading material.")
        else:
            sentences.append(f"Sentence {i} " + "content " * int(rng.integers(3, 10)))

    query = rng.normal(size=dim).astype(np.float32)
    query /= np.linalg.norm(query)

    params = {
        "threshold": float(rng.uniform(0.3, 0.95)),
        "skip_threshold": float(rng.uniform(0.3, 0.95)),
        "min_window": int(rng.integers(1, 4)),
        "max_expand": int(rng.integers(3, 8)),
        "relevance_threshold_pct": float(rng.uniform(0.5, 0.99)),
        "merge_gap": int(rng.integers(1, 4)),
        "target_clusters": int(rng.integers(3, 7)),
        "adaptive_threshold_enabled": bool(rng.random() < 0.5),
    }
    return sentences, embeddings, query, params


@pytest.mark.parametrize("seed", range(20))
def test_adapter_matches_core_on_random_corpora(seed):
    rng = np.random.default_rng(seed)
    sentences, embeddings, query, params = _random_case(rng)
    num_sentences = len(sentences)

    # HPO-style arrays, built exactly as stand/tune.py builds the corpus
    neighbor_sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    sentence_sims = embeddings @ query
    order = np.argsort(sentence_sims)[::-1].astype(np.int32)

    core = DynamicExpansionCore(
        neighbor_sims=neighbor_sims,
        sentence_sims=sentence_sims,
        top_k_indices=order,
        garbage_mask=build_garbage_mask(sentences),
        **params,
    )
    expected = core.expand_and_retrieve()

    # Adapter path: seeds are what a vector retriever would return (top by
    # cosine over the same embeddings, rank order)
    expander, nodes = _make_expander(
        sentences, embeddings, query_embedding=query, **params
    )
    prefetch_k = min(10, num_sentences)
    seeds = [
        NodeWithScore(node=nodes[int(pos)], score=float(sentence_sims[pos]))
        for pos in order[:prefetch_k]
    ]
    result = expander._postprocess_nodes(seeds, None)

    assert len(result) == len(expected)
    for node_score, cluster in zip(result, expected, strict=True):
        positions = _source_positions(node_score)
        assert positions == list(range(cluster.start_idx, cluster.end_idx + 1))
        assert node_score.score == pytest.approx(cluster.score, abs=1e-5)
        assert node_score.node.text == " ".join(
            sentences[cluster.start_idx : cluster.end_idx + 1]
        )
