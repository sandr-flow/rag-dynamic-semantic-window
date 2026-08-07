"""Contract tests for the shared embedding disk cache (step 2.2)."""

import numpy as np
import pytest

from src.embedding_cache import CachingEmbedding, EmbeddingStore, embedding_cache_enabled


class FakeModel:
    """Duck-typed inner model with distinct, recordable vectors per input."""

    model_name = "fake"
    embed_batch_size = 64

    def __init__(self):
        self.text_batches: list[list[str]] = []
        self.query_calls: list[str] = []

    def get_text_embedding_batch(self, texts, **kwargs):
        self.text_batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    def get_query_embedding(self, query):
        self.query_calls.append(query)
        return [float(len(query)), 2.0]


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "fake_cache.pkl"


def test_repeated_batch_is_served_from_cache(store_path):
    inner = FakeModel()
    model = CachingEmbedding(inner=inner, store=EmbeddingStore(store_path))
    texts = ["alpha", "beta beta", "gamma gamma gamma"]

    first = model.get_text_embedding_batch(texts)
    second = model.get_text_embedding_batch(texts)

    assert len(inner.text_batches) == 1
    assert np.allclose(first, second)
    assert first[1][0] == float(len("beta beta"))


def test_partial_hit_embeds_only_misses_in_order(store_path):
    inner = FakeModel()
    model = CachingEmbedding(inner=inner, store=EmbeddingStore(store_path))

    model.get_text_embedding_batch(["alpha", "beta"])
    result = model.get_text_embedding_batch(["alpha", "fresh text", "beta"])

    assert inner.text_batches[1] == ["fresh text"]
    assert [row[0] for row in result] == [
        float(len("alpha")),
        float(len("fresh text")),
        float(len("beta")),
    ]


def test_query_and_text_embeddings_are_namespaced(store_path):
    inner = FakeModel()
    model = CachingEmbedding(inner=inner, store=EmbeddingStore(store_path))

    as_text = model.get_text_embedding("same string")
    as_query = model.get_query_embedding("same string")

    # Both modes computed (no cross-namespace hit) and cached independently.
    assert as_text[1] == 1.0
    assert as_query[1] == 2.0
    assert model.get_query_embedding("same string")[1] == 2.0
    assert len(inner.query_calls) == 1


def test_prewarm_queries_batches_misses_and_fills_query_cache(store_path):
    inner = FakeModel()
    model = CachingEmbedding(inner=inner, store=EmbeddingStore(store_path))
    model.get_query_embedding("warm")

    missed = model.prewarm_queries(["warm", "cold one", "cold two", "cold one"])

    assert missed == 2
    assert inner.query_calls == ["warm", "cold one", "cold two"]
    # Subsequent retrieval-time lookups are pure cache hits.
    model.get_query_embedding("cold one")
    model.get_query_embedding("cold two")
    assert inner.query_calls == ["warm", "cold one", "cold two"]
    assert model.prewarm_queries(["warm", "cold one"]) == 0


def test_cache_persists_across_store_instances(store_path):
    warm_inner = FakeModel()
    warm = CachingEmbedding(inner=warm_inner, store=EmbeddingStore(store_path))
    original = warm.get_text_embedding_batch(["alpha", "beta"])
    warm.flush_cache()

    cold_inner = FakeModel()
    cold = CachingEmbedding(inner=cold_inner, store=EmbeddingStore(store_path))
    reloaded = cold.get_text_embedding_batch(["alpha", "beta"])

    assert cold_inner.text_batches == []
    assert np.allclose(original, reloaded)
    assert cold.cache_stats == {"hits": 2, "misses": 0, "stored": 2}


def test_unreadable_cache_file_is_rebuilt(store_path):
    store_path.write_bytes(b"not a pickle")
    inner = FakeModel()
    model = CachingEmbedding(inner=inner, store=EmbeddingStore(store_path))

    with pytest.warns(UserWarning, match="unreadable"):
        result = model.get_text_embedding_batch(["alpha"])

    assert result[0][0] == float(len("alpha"))
    assert len(inner.text_batches) == 1


def test_embedding_cache_enabled_env_switch(monkeypatch):
    monkeypatch.delenv("EMBEDDING_CACHE", raising=False)
    assert embedding_cache_enabled()
    for value in ("0", "false", "OFF"):
        monkeypatch.setenv("EMBEDDING_CACHE", value)
        assert not embedding_cache_enabled()
    monkeypatch.setenv("EMBEDDING_CACHE", "1")
    assert embedding_cache_enabled()
