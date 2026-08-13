"""Content-addressed disk cache for embedding vectors.

``CachingEmbedding`` wraps any LlamaIndex embedding model and memoizes every
vector on disk, keyed by a hash of the exact input text. Because the key is
the text itself (not a dataset or run id), reuse is automatic wherever the
same string is embedded again:

- repeated ``stand run`` sweeps over the same dataset re-embed nothing —
  expansion parameters never touch the index;
- ``stand tune`` and ``stand run`` share sentence vectors, since both build
  identical phantom texts via ``build_embedding_texts``;
- one question asked to several strategies is embedded once per run.

Query embeddings are namespaced separately from text embeddings: models such
as bge prepend a query instruction, so the same string yields different
vectors in the two modes.

Storage is a directory of append-only shards per (provider, model): each
flush writes only the vectors accumulated since the previous one, as a
``shard-*.npy`` matrix plus a ``shard-*.json`` key list. Rewriting the whole
store on every flush (the previous single-pickle layout) cost O(store) per
flush and therefore O(store x new vectors) per run — 118s per flush once the
FinanceBench corpus pushed the store past 6 GB. Appending costs O(new).

Shards are read back through ``mmap``, so a warm run keeps only the key
index in RAM (~120 MB per million vectors) instead of every vector. Once
``compact_every`` shards accumulate they are streamed into a single shard so
the directory does not grow without bound. A legacy single-pickle store is
migrated into the shard layout on first load.

Vectors are stored as float32 (the pipeline dtype); writes are atomic
(temp file + ``os.replace``).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from pydantic import PrivateAttr


def _cache_key(kind: str, text: str) -> str:
    return hashlib.sha256(f"{kind}\x00{text}".encode()).hexdigest()


class EmbeddingStore:
    """Append-only shard store with lazy mmap load and atomic flushes."""

    def __init__(
        self,
        path: Path | str,
        *,
        flush_every: int = 4096,
        compact_every: int = 32,
    ):
        # ``path`` names the legacy pickle; the shard directory sits beside it
        # under the same stem so existing call sites need no change.
        self._legacy_path = Path(path)
        self._dir = self._legacy_path.with_suffix("")
        self._flush_every = flush_every
        self._compact_every = compact_every
        # key -> (shard index, row); vectors stay on disk behind mmap.
        self._index: dict[str, tuple[int, int]] | None = None
        self._shards: list[np.ndarray] = []
        self._pending: dict[str, np.ndarray] = {}
        self._seq = 0
        self.hits = 0
        self.misses = 0
        # Backstop for hard failures mid-run; normal exits flush explicitly.
        atexit.register(self.flush)

    # -- layout helpers ----------------------------------------------------

    def _shard_stems(self) -> list[Path]:
        """Shard stems with both parts present, in creation order."""
        if not self._dir.exists():
            return []
        # Oldest first: within a run the index is filled in flush order, so
        # replaying shards by age keeps the same last-writer-wins semantics.
        # (Keys are content hashes, so duplicates hold equivalent vectors —
        # order affects which copy is read, never what it means.)
        found = sorted(self._dir.glob("shard-*.json"), key=lambda p: (p.stat().st_mtime, p.name))
        stems = [p.with_suffix("") for p in found]
        return [stem for stem in stems if stem.with_suffix(".npy").exists()]

    def _commit_shard(self, stem: Path, keys: list[str]) -> Path:
        """Publish a staged ``.npy.tmp``; the .json lands last as the marker."""
        os.replace(stem.with_suffix(".npy.tmp"), stem.with_suffix(".npy"))
        json_tmp = stem.with_suffix(".json.tmp")
        json_tmp.write_text(json.dumps(keys), encoding="utf-8")
        os.replace(json_tmp, stem.with_suffix(".json"))
        return stem

    def _write_shard(self, keys: list[str], vectors: np.ndarray, name: str) -> Path:
        """Write one shard atomically from an in-memory matrix."""
        self._dir.mkdir(parents=True, exist_ok=True)
        stem = self._dir / name
        # Save through a handle: np.save would append another .npy to a path
        # whose suffix is not exactly that.
        with open(stem.with_suffix(".npy.tmp"), "wb") as f:
            np.save(f, vectors, allow_pickle=False)
        return self._commit_shard(stem, keys)

    def _attach_shard(self, stem: Path) -> None:
        """Register an on-disk shard into the in-memory key index."""
        keys = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        vectors = np.load(stem.with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
        shard_idx = len(self._shards)
        self._shards.append(vectors)
        if self._index is None:
            self._index = {}
        for row, key in enumerate(keys):
            self._index[key] = (shard_idx, row)

    # -- load / migrate ----------------------------------------------------

    def _migrate_legacy(self) -> None:
        """Fold a single-pickle store into the shard layout, once."""
        try:
            with open(self._legacy_path, "rb") as f:
                entries = pickle.load(f)
        except Exception as exc:
            warnings.warn(
                f"Embedding cache {self._legacy_path} is unreadable ({exc!r}); rebuilding.",
                stacklevel=3,
            )
            self._legacy_path.unlink(missing_ok=True)
            return
        if not isinstance(entries, dict):
            warnings.warn(
                f"Embedding cache {self._legacy_path} is unreadable "
                f"(expected a dict, got {type(entries).__name__}); rebuilding.",
                stacklevel=3,
            )
            self._legacy_path.unlink(missing_ok=True)
            return

        started = time.time()
        keys = list(entries)
        if keys:
            # Stream row by row into a memmap, dropping each source vector as
            # it lands: stacking first would hold two copies of a
            # multi-gigabyte store in RAM at once.
            self._dir.mkdir(parents=True, exist_ok=True)
            stem = self._dir / "shard-migrated-0000"
            width = len(np.asarray(entries[keys[0]]))
            merged = np.lib.format.open_memmap(
                stem.with_suffix(".npy.tmp"),
                mode="w+",
                dtype=np.float32,
                shape=(len(keys), width),
            )
            for row, key in enumerate(keys):
                merged[row] = np.asarray(entries.pop(key), dtype=np.float32)
            merged.flush()
            del merged
            self._commit_shard(stem, keys)
        del entries
        # The shard is a complete copy; keeping the pickle would double a
        # multi-gigabyte store on disk for no benefit.
        self._legacy_path.unlink(missing_ok=True)
        print(
            f"[embedding-cache] migrated {len(keys)} vectors from "
            f"{self._legacy_path.name} into shards in {time.time() - started:.1f}s",
            flush=True,
        )

    def _load(self) -> dict[str, tuple[int, int]]:
        if self._index is None:
            started = time.time()
            if self._legacy_path.exists():
                self._migrate_legacy()
            self._index = {}
            self._shards = []
            for stem in self._shard_stems():
                self._attach_shard(stem)
            print(
                f"[embedding-cache] loaded {len(self._index)} vectors from "
                f"{len(self._shards)} shard(s) in {self._dir} "
                f"in {time.time() - started:.1f}s",
                flush=True,
            )
        return self._index

    # -- public API --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._load()) + len(self._pending)

    def get(self, key: str) -> np.ndarray | None:
        index = self._load()
        pending = self._pending.get(key)
        if pending is not None:
            self.hits += 1
            return pending
        location = index.get(key)
        if location is None:
            self.misses += 1
            return None
        self.hits += 1
        shard_idx, row = location
        return np.asarray(self._shards[shard_idx][row])

    def put(self, key: str, vector: Any) -> None:
        self._load()
        self._pending[key] = np.asarray(vector, dtype=np.float32)
        if len(self._pending) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        started = time.time()
        keys = list(self._pending)
        vectors = np.stack([self._pending[k] for k in keys])
        self._dir.mkdir(parents=True, exist_ok=True)
        while True:
            self._seq += 1
            name = f"shard-{os.getpid()}-{self._seq:04d}"
            stem = self._dir / name
            if not stem.with_suffix(".npy").exists() and not stem.with_suffix(".json").exists():
                break
        stem = self._write_shard(keys, vectors, name)
        self._pending = {}
        self._attach_shard(stem)
        print(
            f"[embedding-cache] appended {len(keys)} new vectors "
            f"({len(self._index or {})} stored) to {stem.name} "
            f"in {time.time() - started:.1f}s",
            flush=True,
        )
        if len(self._shards) >= self._compact_every:
            self.compact()

    def compact(self) -> None:
        """Merge every shard into one, streaming rows to keep RAM flat."""
        stems = self._shard_stems()
        if len(stems) < 2:
            return
        started = time.time()
        index = self._load()
        # Deduplicate: a later shard wins, matching lookup order.
        rows = [(key, shard_idx, row) for key, (shard_idx, row) in index.items()]
        if not rows:
            return
        width = self._shards[rows[0][1]].shape[1]
        self._dir.mkdir(parents=True, exist_ok=True)
        target = self._dir / "shard-compact-0000"
        npy_tmp = target.with_suffix(".npy.tmp")
        merged = np.lib.format.open_memmap(
            npy_tmp, mode="w+", dtype=np.float32, shape=(len(rows), width)
        )
        keys: list[str] = []
        for new_row, (key, shard_idx, row) in enumerate(rows):
            merged[new_row] = self._shards[shard_idx][row]
            keys.append(key)
        merged.flush()
        del merged
        # Drop mmap handles before replacing files underneath them (Windows
        # refuses to unlink a mapped file).
        self._shards = []
        self._index = None
        os.replace(npy_tmp, target.with_suffix(".npy"))
        json_tmp = target.with_suffix(".json.tmp")
        json_tmp.write_text(json.dumps(keys), encoding="utf-8")
        os.replace(json_tmp, target.with_suffix(".json"))
        for stem in stems:
            if stem != target:
                stem.with_suffix(".npy").unlink(missing_ok=True)
                stem.with_suffix(".json").unlink(missing_ok=True)
        self._index = {}
        self._attach_shard(target)
        print(
            f"[embedding-cache] compacted {len(stems)} shards into "
            f"{len(keys)} vectors in {time.time() - started:.1f}s",
            flush=True,
        )


class CachingEmbedding(BaseEmbedding):
    """Wrap an embedding model with an :class:`EmbeddingStore`.

    Batch requests are served from the cache per text; only misses reach the
    inner model, in a single batched call. The wrapper mirrors the inner
    model's ``embed_batch_size`` so upstream chunking is unchanged.
    """

    _inner: Any = PrivateAttr()
    _store: EmbeddingStore = PrivateAttr()

    def __init__(self, inner: Any, store: EmbeddingStore, **kwargs: Any):
        super().__init__(
            model_name=getattr(inner, "model_name", "unknown"),
            embed_batch_size=getattr(inner, "embed_batch_size", 64),
            **kwargs,
        )
        self._inner = inner
        self._store = store

    @classmethod
    def class_name(cls) -> str:
        return "CachingEmbedding"

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._store.hits,
            "misses": self._store.misses,
            "stored": len(self._store),
        }

    def flush_cache(self) -> None:
        self._store.flush()

    def prewarm_queries(self, queries: list[str]) -> int:
        """Batch-embed uncached queries ahead of retrieval; returns miss count.

        There is no public batch API for query-mode embeddings in LlamaIndex,
        so during evaluation every question would otherwise cost one blocking
        request. This issues the misses concurrently (embed_batch_size at a
        time) through the inner model's query path, preserving query-mode
        semantics for instruction-prefixed models such as bge.
        """
        unique = list(dict.fromkeys(queries))
        missing = [
            q for q in unique if self._store.get(_cache_key("query", q)) is None
        ]
        if not missing:
            return 0

        started = time.time()
        print(
            f"[embedding-cache] query prewarm: {len(unique) - len(missing)} hits, "
            f"{len(missing)} misses; requesting {len(missing)} embeddings",
            flush=True,
        )

        aget = getattr(self._inner, "aget_query_embedding", None)
        if aget is None:
            embeddings = [self._inner.get_query_embedding(q) for q in missing]
        else:
            import asyncio

            async def _gather() -> list[Embedding]:
                out: list[Embedding] = []
                for start in range(0, len(missing), self.embed_batch_size):
                    chunk = missing[start : start + self.embed_batch_size]
                    out.extend(await asyncio.gather(*(aget(q) for q in chunk)))
                return out

            embeddings = asyncio.run(_gather())

        for query, embedding in zip(missing, embeddings, strict=True):
            self._store.put(_cache_key("query", query), embedding)
        print(
            f"[embedding-cache] query prewarm received {len(missing)} embeddings "
            f"in {time.time() - started:.1f}s",
            flush=True,
        )
        return len(missing)

    def _get_query_embedding(self, query: str) -> Embedding:
        cached = self._store.get(_cache_key("query", query))
        if cached is not None:
            return cached.tolist()
        started = time.time()
        print("[embedding-cache] query miss: requesting 1 embedding", flush=True)
        embedding = self._inner.get_query_embedding(query)
        self._store.put(_cache_key("query", query), embedding)
        print(
            f"[embedding-cache] query embedding received in {time.time() - started:.1f}s",
            flush=True,
        )
        return embedding

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        results: list[Embedding | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            cached = self._store.get(_cache_key("text", text))
            if cached is None:
                missing.append(i)
            else:
                results[i] = cached.tolist()
        if missing:
            started = time.time()
            print(
                f"[embedding-cache] text batch: {len(texts) - len(missing)} hits, "
                f"{len(missing)} misses; requesting {len(missing)} embeddings",
                flush=True,
            )
            fresh = self._inner.get_text_embedding_batch([texts[i] for i in missing])
            for i, embedding in zip(missing, fresh, strict=True):
                self._store.put(_cache_key("text", texts[i]), embedding)
                results[i] = embedding
            print(
                f"[embedding-cache] text batch received {len(missing)} embeddings "
                f"in {time.time() - started:.1f}s",
                flush=True,
            )
        return results  # type: ignore[return-value]


def embedding_cache_enabled() -> bool:
    """Disk cache is on unless EMBEDDING_CACHE is set to 0/false/off."""
    return os.getenv("EMBEDDING_CACHE", "1").strip().lower() not in {"0", "false", "off"}
