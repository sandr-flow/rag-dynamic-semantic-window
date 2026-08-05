"""The single configuration object that drives one benchmark run.

One run = one combination (one dataset, one embedding model, one strategy set,
one index mode). Matrices are done by running several times and comparing
saved results. Every knob is declared here exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INDEX_MODES = ("per_document", "shared")
PARAM_SOURCES = ("default", "tuned")

DEFAULT_STRATEGIES = [
    "naive",
    "fixed_window",
    "token_text",
    "semantic_splitter",
    "dynamic_semantic",
]


@dataclass
class RunConfig:
    """A fully resolved single-combination benchmark request."""

    dataset: str
    """Name of a prepared dataset artifact."""

    embedding: str = "mock"
    """Name of an embedding model from the registry (or a builtin)."""

    strategies: list[str] = field(default_factory=lambda: list(DEFAULT_STRATEGIES))
    """Strategy ids to compare. dynamic_semantic is the strategy under test."""

    index_mode: str = "shared"
    """shared = one corpus-wide index (primary mode); per_document = one
    collection per doc, a diagnostic isolating chunking behavior."""

    params: str = "default"
    """default = config.py defaults; tuned = load tuned artifact for (dataset, embedding)."""

    tuned_dataset: str | None = None
    """When set with params=tuned, load tuned artifact from this dataset instead
    of ``dataset`` (same embedding). Useful for cross-eval, e.g. extrahard
    tuned params on the underlying hard dataset."""

    top_k: int = 5
    """Chunks/clusters retrieved per query."""

    metric_k: int | None = None
    """Cutoff for HR/P/R/NDCG. Defaults to top_k."""

    limit: int | None = None
    """Optional cap on number of documents from the dataset."""

    dynamic_overrides: dict[str, Any] = field(default_factory=dict)
    """dynamic_semantic param overrides applied on top of default/tuned params
    (ablations and manual sweeps). Keys are validated by the runner."""

    def __post_init__(self) -> None:
        if self.index_mode not in INDEX_MODES:
            raise ValueError(
                f"index_mode must be one of {INDEX_MODES}, got {self.index_mode!r}"
            )
        if self.params not in PARAM_SOURCES:
            raise ValueError(
                f"params must be one of {PARAM_SOURCES}, got {self.params!r}"
            )
        if not self.strategies:
            raise ValueError("strategies must not be empty")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

    @property
    def effective_metric_k(self) -> int:
        return self.metric_k or self.top_k

    @classmethod
    def from_dict(cls, data: dict) -> RunConfig:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
