"""Reusable artifact registry: datasets, embedding models, tuned params.

Everything the menu offers to choose from lives here. Prep scripts write
artifacts; the runner and menu only read them.

Layout::

    artifacts/
      datasets/<name>/manifest.json + data.jsonl
      tuned/<dataset>__<embedding>/manifest.json
      embeddings.json
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.benchmark_datasets import load_benchmark_dataset

from . import paths

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class DatasetInfo:
    """Summary of a prepared dataset artifact (from its manifest)."""

    name: str
    source: str
    qa_model: str
    num_items: int
    num_questions: int
    created_at: str

    def label(self) -> str:
        return (
            f"{self.name}  ({self.source}, qa={self.qa_model}, "
            f"{self.num_items} docs / {self.num_questions} q)"
        )


def slugify(value: str) -> str:
    """Make a filesystem- and key-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower()).strip("_")
    return slug or "unnamed"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def dataset_dir(name: str) -> Path:
    return paths.DATASETS_DIR / slugify(name)


def save_dataset(
    name: str,
    items: list[dict[str, Any]],
    *,
    source: str,
    qa_model: str,
) -> DatasetInfo:
    """Write a dataset artifact (data.jsonl + manifest.json) and return its info."""
    target = dataset_dir(name)
    target.mkdir(parents=True, exist_ok=True)

    with open(target / "data.jsonl", "w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            record = {
                "id": item.get("id", idx),
                "title": item.get("title", f"item_{idx}"),
                "text": item["text"],
                "qa_pairs": item.get("qa_pairs", []),
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    num_questions = sum(len(item.get("qa_pairs", [])) for item in items)
    manifest = {
        "name": slugify(name),
        "source": source,
        "qa_model": qa_model,
        "num_items": len(items),
        "num_questions": num_questions,
        "created_at": _now(),
    }
    with open(target / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return DatasetInfo(**manifest)


def list_datasets() -> list[DatasetInfo]:
    """Return all prepared datasets, newest first."""
    if not paths.DATASETS_DIR.exists():
        return []
    infos: list[DatasetInfo] = []
    for manifest_path in paths.DATASETS_DIR.glob("*/manifest.json"):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            infos.append(DatasetInfo(**{k: data[k] for k in DatasetInfo.__annotations__}))
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return sorted(infos, key=lambda info: info.created_at, reverse=True)


def get_dataset(name: str) -> DatasetInfo | None:
    manifest_path = dataset_dir(name) / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return DatasetInfo(**{k: data[k] for k in DatasetInfo.__annotations__})


def load_dataset_items(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load benchmark items (title/text/qa_pairs) from a prepared dataset."""
    data_path = dataset_dir(name) / "data.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found. Prepare it first: "
            f"python -m stand prepare-dataset ..."
        )
    items = load_benchmark_dataset(dataset_path=str(data_path))
    return items[:limit] if limit else items


# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingInfo:
    """A selectable embedding model."""

    name: str
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    builtin: bool = False

    def label(self) -> str:
        tag = " [builtin]" if self.builtin else ""
        return f"{self.name}  ({self.provider}/{self.model}){tag}"


# Always available without a prep step: offline mock + the default local HF model.
_BUILTIN_EMBEDDINGS = [
    EmbeddingInfo(name="mock", provider="mock", model="mock:384", builtin=True),
    EmbeddingInfo(
        name="bge-small",
        provider="huggingface",
        model="BAAI/bge-small-en-v1.5",
        builtin=True,
    ),
]


def list_embeddings() -> list[EmbeddingInfo]:
    """Return registered embedding models plus always-available builtins."""
    registered: dict[str, EmbeddingInfo] = {}
    if paths.EMBEDDINGS_REGISTRY.exists():
        with open(paths.EMBEDDINGS_REGISTRY, encoding="utf-8") as f:
            for entry in json.load(f):
                info = EmbeddingInfo(**entry)
                registered[info.name] = info

    result = {info.name: info for info in _BUILTIN_EMBEDDINGS}
    result.update(registered)  # registered entries may override a builtin name
    return list(result.values())


def get_embedding(name: str) -> EmbeddingInfo | None:
    for info in list_embeddings():
        if info.name == name:
            return info
    return None


def register_embedding(info: EmbeddingInfo) -> None:
    """Add or update an embedding registration."""
    paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    if paths.EMBEDDINGS_REGISTRY.exists():
        with open(paths.EMBEDDINGS_REGISTRY, encoding="utf-8") as f:
            entries = [e for e in json.load(f) if e.get("name") != info.name]
    entries.append({k: v for k, v in asdict(info).items() if k != "builtin"})
    with open(paths.EMBEDDINGS_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tuned hyperparameters (per dataset + embedding — the per-domain hypothesis)
# ---------------------------------------------------------------------------


def tuned_key(dataset: str, embedding: str) -> str:
    return f"{slugify(dataset)}__{slugify(embedding)}"


def tuned_dir(dataset: str, embedding: str) -> Path:
    return paths.TUNED_DIR / tuned_key(dataset, embedding)


def save_tuned(
    dataset: str,
    embedding: str,
    params: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    *,
    metrics_train: dict[str, Any] | None = None,
    metrics_val: dict[str, Any] | None = None,
    tuning: dict[str, Any] | None = None,
) -> Path:
    """Persist Optuna best params for a (dataset, embedding) domain."""
    target = tuned_dir(dataset, embedding)
    target.mkdir(parents=True, exist_ok=True)
    resolved_metrics = metrics or metrics_val or metrics_train or {}
    payload = {
        "dataset": slugify(dataset),
        "embedding": slugify(embedding),
        "params": params,
        "metrics": resolved_metrics,
        "created_at": _now(),
    }
    if metrics_train is not None:
        payload["metrics_train"] = metrics_train
    if metrics_val is not None:
        payload["metrics_val"] = metrics_val
    if tuning is not None:
        payload["tuning"] = tuning
    with open(target / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return target


def load_tuned(dataset: str, embedding: str) -> dict[str, Any] | None:
    """Return tuned dynamic_semantic params for a domain, or None."""
    manifest_path = tuned_dir(dataset, embedding) / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def has_tuned(dataset: str, embedding: str) -> bool:
    return (tuned_dir(dataset, embedding) / "manifest.json").exists()
