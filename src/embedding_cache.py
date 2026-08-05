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

Vectors are stored as float32 (the pipeline dtype). One store file per
(provider, model); writes are batched and atomic (temp file + ``os.replace``).
"""

from __future__ import annotations

import atexit
import hashlib
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
    """Dict-on-disk vector store with lazy load and batched atomic flushes."""

    def __init__(self, path: Path | str, *, flush_every: int = 4096):
        self._path = Path(path)
        self._flush_every = flush_every
        self._entries: dict[str, np.ndarray] | None = None
        self._dirty = 0
        self.hits = 0
        self.misses = 0
        # Backstop for hard failures mid-run; normal exits flush explicitly.
        atexit.register(self.flush)

    def _load(self) -> dict[str, np.ndarray]:
        if self._entries is None:
            started = time.time()
            if self._path.exists():
                try:
                    with open(self._path, "rb") as f:
                        self._entries = pickle.load(f)
                except Exception as exc:
                    warnings.warn(
                        f"Embedding cache {self._path} is unreadable ({exc!r}); rebuilding.",
                        stacklevel=2,
                    )
                    self._entries = {}
            else:
                self._entries = {}
            print(
                f"[embedding-cache] loaded {len(self._entries)} vectors "
                f"from {self._path} in {time.time() - started:.1f}s",
                flush=True,
            )
        return self._entries

    def __len__(self) -> int:
        return len(self._load())

    def get(self, key: str) -> np.ndarray | None:
        vector = self._load().get(key)
        if vector is None:
            self.misses += 1
        else:
            self.hits += 1
        return vector

    def put(self, key: str, vector: Any) -> None:
        self._load()[key] = np.asarray(vector, dtype=np.float32)
        self._dirty += 1
        if self._dirty >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._dirty or self._entries is None:
            return
        started = time.time()
        dirty = self._dirty
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_name(f"{self._path.name}.{os.getpid()}.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(self._entries, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self._path)
        self._dirty = 0
        print(
            f"[embedding-cache] flushed {dirty} new vectors "
            f"({len(self._entries)} stored) to {self._path} "
            f"in {time.time() - started:.1f}s",
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
