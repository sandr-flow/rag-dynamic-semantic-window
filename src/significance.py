"""Paired bootstrap significance testing for per-question benchmark metrics.

The benchmark evaluates every strategy on the same questions in the same
order, so per-question metric rows are paired. For a target strategy vs a
baseline, the statistic is the mean per-question difference; the 95% CI
comes from resampling questions with replacement. A difference is called
significant when the CI excludes zero.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


@dataclass
class PairedComparison:
    """One target-vs-baseline delta for one metric."""

    target: str
    baseline: str
    metric: str
    n: int
    delta: float
    ci_low: float
    ci_high: float
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_bootstrap_ci(
    target_values,
    baseline_values,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Mean paired difference and its bootstrap CI.

    Returns ``(delta, ci_low, ci_high)`` where ``delta`` is
    ``mean(target - baseline)`` over questions.
    """
    a = np.asarray(target_values, dtype=np.float64)
    b = np.asarray(baseline_values, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(
            f"paired arrays must be 1-D and equal-length, got {a.shape} vs {b.shape}"
        )
    if len(a) == 0:
        raise ValueError("paired arrays must not be empty")

    diffs = a - b
    delta = float(diffs.mean())

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    resampled_means = diffs[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(resampled_means, [alpha, 1.0 - alpha])
    return delta, float(ci_low), float(ci_high)


def compare_to_baselines(
    aggregate: dict[str, list[dict]],
    *,
    target: str,
    metric_keys: list[str],
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> list[PairedComparison]:
    """Compare the target strategy against every other strategy in aggregate.

    Strategies whose row count differs from the target's (e.g. a strategy
    that failed to build on part of the documents) cannot be paired and are
    skipped with a warning.
    """
    target_rows = aggregate.get(target)
    if not target_rows:
        return []

    comparisons: list[PairedComparison] = []
    for baseline, rows in aggregate.items():
        if baseline == target:
            continue
        if len(rows) != len(target_rows):
            warnings.warn(
                f"skipping '{baseline}': {len(rows)} rows vs {len(target_rows)} "
                f"for '{target}' - not paired",
                stacklevel=2,
            )
            continue
        for metric in metric_keys:
            delta, ci_low, ci_high = paired_bootstrap_ci(
                [row[metric] for row in target_rows],
                [row[metric] for row in rows],
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
            )
            comparisons.append(
                PairedComparison(
                    target=target,
                    baseline=baseline,
                    metric=metric,
                    n=len(target_rows),
                    delta=delta,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    significant=not (ci_low <= 0.0 <= ci_high),
                )
            )
    return comparisons


def format_comparisons(
    comparisons: list[PairedComparison],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[str]:
    """Human-readable table lines; 'ns' marks non-significant differences."""
    if not comparisons:
        return []

    target = comparisons[0].target
    metrics: list[str] = []
    by_baseline: dict[str, dict[str, PairedComparison]] = {}
    for comparison in comparisons:
        by_baseline.setdefault(comparison.baseline, {})[comparison.metric] = comparison
        if comparison.metric not in metrics:
            metrics.append(comparison.metric)

    pct = int(round(confidence * 100))
    cell_width = 30
    lines = [
        f"Paired bootstrap vs {target} "
        f"({n_resamples} resamples, {pct}% CI; ns = not significant):"
    ]
    header = f"{'Baseline':20} | " + " | ".join(
        f"{'d' + metric:>{cell_width}}" for metric in metrics
    )
    lines.append("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    for baseline, per_metric in by_baseline.items():
        cells = []
        for metric in metrics:
            comparison = per_metric.get(metric)
            if comparison is None:
                cells.append(f"{'-':>{cell_width}}")
                continue
            marker = "" if comparison.significant else " ns"
            cell = (
                f"{comparison.delta:+.4f} "
                f"[{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]{marker}"
            )
            cells.append(f"{cell:>{cell_width}}")
        lines.append(f"{baseline:20} | " + " | ".join(cells))
    return lines
