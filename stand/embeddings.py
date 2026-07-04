"""Embedding model construction for the stand.

One place that resolves a registered embedding name, builds the provider
model, wraps it with the shared disk cache (step 2.2) and installs it into
``Settings``. Both ``stand run`` and ``stand tune`` go through here, so they
share one vector store per (provider, model).
"""

from __future__ import annotations

from llama_index.core import Settings

from src.config import EmbeddingProviderConfig
from src.embedding_cache import CachingEmbedding, EmbeddingStore, embedding_cache_enabled
from src.providers import build_embedding_model, embedding_config_from_env

from . import artifacts, paths


def prepare_embed_model(embedding: str):
    """Build (and cache-wrap) the embedding model; set ``Settings.embed_model``.

    Returns ``(embed_model, embedding_config)``.
    """
    info = artifacts.get_embedding(embedding)
    if info is None:
        raise ValueError(f"Unknown embedding '{embedding}'. See: python -m stand list")
    config = embedding_config_from_env(
        provider=info.provider,
        model=info.model,
        api_key_env=info.api_key_env,
        base_url=info.base_url,
    )
    model = build_embedding_model(config)
    if embedding_cache_enabled():
        model = CachingEmbedding(inner=model, store=EmbeddingStore(_store_path(config)))
    Settings.embed_model = model
    return model, config


def _store_path(config: EmbeddingProviderConfig):
    key = f"{artifacts.slugify(config.provider)}__{artifacts.slugify(config.model)}"
    return paths.EMBEDDING_CACHE_DIR / f"{key}.pkl"


def report_cache(embed_model, *, verbose: bool = True) -> None:
    """Flush the disk cache and optionally print hit/miss stats."""
    if not isinstance(embed_model, CachingEmbedding):
        return
    embed_model.flush_cache()
    if verbose:
        stats = embed_model.cache_stats
        print(
            f"[INFO] embedding cache: {stats['hits']} hits / {stats['misses']} misses "
            f"({stats['stored']} vectors on disk)"
        )
