"""Contract tests for the shared embedding disk cache (step 2.2)."""

import pickle

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


def test_flush_appends_a_shard_and_leaves_earlier_ones_untouched(store_path):
    """The whole point of the shard layout: a flush costs O(new vectors)."""
    store = EmbeddingStore(store_path, flush_every=1000)
    store.put("k1", [1.0, 2.0])
    store.flush()

    shards = sorted((store_path.with_suffix("")).glob("shard-*.npy"))
    assert len(shards) == 1
    first_bytes = shards[0].read_bytes()

    store.put("k2", [3.0, 4.0])
    store.flush()

    shards_after = sorted((store_path.with_suffix("")).glob("shard-*.npy"))
    assert len(shards_after) == 2
    # The pre-existing shard was not rewritten — that is what used to make a
    # flush cost O(entire store).
    assert shards_after[0].read_bytes() == first_bytes
    assert np.allclose(store.get("k1"), [1.0, 2.0])
    assert np.allclose(store.get("k2"), [3.0, 4.0])


def test_legacy_pickle_store_is_migrated_into_shards(store_path):
    legacy = {"old_key": np.array([7.0, 8.0], dtype=np.float32)}
    with open(store_path, "wb") as f:
        pickle.dump(legacy, f)

    store = EmbeddingStore(store_path)
    assert np.allclose(store.get("old_key"), [7.0, 8.0])
    # The shard copy is complete, so the pickle is not kept around to double
    # a multi-gigabyte store on disk.
    assert not store_path.exists()
    assert list((store_path.with_suffix("")).glob("shard-*.npy"))

    # A fresh store instance reads the migrated vectors back.
    assert np.allclose(EmbeddingStore(store_path).get("old_key"), [7.0, 8.0])


def test_compaction_merges_shards_and_preserves_every_key(store_path):
    store = EmbeddingStore(store_path, flush_every=1, compact_every=3)
    for i in range(6):
        store.put(f"k{i}", [float(i), 1.0])

    shard_dir = store_path.with_suffix("")
    assert len(list(shard_dir.glob("shard-*.npy"))) < 6

    for i in range(6):
        assert np.allclose(store.get(f"k{i}"), [float(i), 1.0]), i
    reopened = EmbeddingStore(store_path)
    assert len(reopened) == 6
    assert np.allclose(reopened.get("k5"), [5.0, 1.0])


def test_embedding_cache_enabled_env_switch(monkeypatch):
    monkeypatch.delenv("EMBEDDING_CACHE", raising=False)
    assert embedding_cache_enabled()
    for value in ("0", "false", "OFF"):
        monkeypatch.setenv("EMBEDDING_CACHE", value)
        assert not embedding_cache_enabled()
    monkeypatch.setenv("EMBEDDING_CACHE", "1")
    assert embedding_cache_enabled()
