"""Filesystem layout for the benchmark stand."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS = ROOT / "artifacts"
DATASETS_DIR = ARTIFACTS / "datasets"
TUNED_DIR = ARTIFACTS / "tuned"
EMBEDDINGS_REGISTRY = ARTIFACTS / "embeddings.json"
CORPUS_CACHE_DIR = ARTIFACTS / "corpus_cache"
EMBEDDING_CACHE_DIR = ARTIFACTS / "embedding_cache"

RESULTS_DIR = ROOT / "results"


def ensure_dirs() -> None:
    """Create the artifact directory tree if missing."""
    for path in (DATASETS_DIR, TUNED_DIR, CORPUS_CACHE_DIR, EMBEDDING_CACHE_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
