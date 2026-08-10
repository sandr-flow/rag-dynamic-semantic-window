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
    kind: str = "standard"
    corpus_dataset: str | None = None

    def label(self) -> str:
        if self.kind == "extrahard" and self.corpus_dataset:
            return (
                f"{self.name}  (extrahard:{self.corpus_dataset}, "
                f"{self.num_questions} compound q, corpus {self.num_items} docs)"
            )
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


def _dataset_info_from_manifest(data: dict[str, Any]) -> DatasetInfo:
    fields = {k: data[k] for k in DatasetInfo.__annotations__ if k in data}
    if "kind" not in fields:
        fields["kind"] = data.get("kind", "standard")
    if "corpus_dataset" not in fields:
        fields["corpus_dataset"] = data.get("corpus_dataset")
    return DatasetInfo(**fields)


def load_manifest(name: str) -> dict[str, Any]:
    """Load the full dataset manifest (including extrahard metadata)."""
    manifest_path = dataset_dir(name) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found. Prepare it first: "
            f"python -m stand prepare-dataset ..."
        )
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def is_extrahard(name: str) -> bool:
    return load_manifest(name).get("kind") == "extrahard"


def load_corpus_items(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load article items to index (corpus) for a dataset or extrahard artifact."""
    manifest = load_manifest(name)
    corpus_name = manifest.get("corpus_dataset") if manifest.get("kind") == "extrahard" else name
    return load_dataset_items(corpus_name, limit=limit)


def load_eval_questions(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load evaluation questions (flat qa list or extrahard compound pairs)."""
    manifest = load_manifest(name)
    if manifest.get("kind") == "extrahard":
        pairs_path = dataset_dir(name) / "pairs.jsonl"
        if not pairs_path.exists():
            raise FileNotFoundError(f"Extrahard dataset '{name}' is missing pairs.jsonl")
        pairs: list[dict[str, Any]] = []
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs[:limit] if limit else pairs

    items = load_dataset_items(name, limit=limit)
    # Tag each question with the document that answers it, in the same
    # ``source_docs`` form extrahard pairs use, so the runner can drop
    # questions whose document never makes it into the index.
    return [
        {**qa, "source_docs": [str(item.get("id", item_idx))]}
        for item_idx, item in enumerate(items)
        for qa in item.get("qa_pairs", [])
    ]


def save_extrahard_dataset(
    name: str,
    pairs: list[dict[str, Any]],
    *,
    source_name: str,
    corpus_dataset: str,
    corpus_num_items: int,
    partners_per_question: int,
    pair_seed: int,
) -> DatasetInfo:
    """Write an extrahard artifact (pairs.jsonl + manifest.json)."""
    target = dataset_dir(name)
    target.mkdir(parents=True, exist_ok=True)

    with open(target / "pairs.jsonl", "w", encoding="utf-8") as f:
        for pair in pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write("\n")

    manifest = {
        "name": slugify(name),
        "kind": "extrahard",
        "source": f"extrahard:{slugify(source_name)}",
        "corpus_dataset": slugify(corpus_dataset),
        "qa_model": "combinatorial",
        "num_items": corpus_num_items,
        "num_questions": len(pairs),
        "pairing": "cross_document",
        "partners_per_question": partners_per_question,
        "pair_seed": pair_seed,
        "created_at": _now(),
    }
    with open(target / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return _dataset_info_from_manifest(manifest)


def list_datasets() -> list[DatasetInfo]:
    """Return all prepared datasets, newest first."""
    if not paths.DATASETS_DIR.exists():
        return []
    infos: list[DatasetInfo] = []
    for manifest_path in paths.DATASETS_DIR.glob("*/manifest.json"):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            infos.append(_dataset_info_from_manifest(data))
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return sorted(infos, key=lambda info: info.created_at, reverse=True)


def get_dataset(name: str) -> DatasetInfo | None:
    manifest_path = dataset_dir(name) / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return _dataset_info_from_manifest(data)


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
