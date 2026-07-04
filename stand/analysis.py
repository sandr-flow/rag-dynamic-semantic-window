"""Post-hoc analysis over saved result JSONs.

Result files store per-question metric rows (``aggregate``), so paired
significance can be computed after the fact — including across two files,
e.g. a tuned dynamic-only run vs an earlier all-baselines run. Cross-file
pairing is valid because runs iterate the same dataset artifact in the same
document/question order; the guard below rejects files whose config makes
rows non-comparable.
"""

from __future__ import annotations

import json

from src.significance import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    PairedComparison,
    compare_to_baselines,
)

DEFAULT_TARGET = "Dynamic Semantic"


def _load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def significance_report(
    result_path: str,
    *,
    baseline_result_path: str | None = None,
    target: str = DEFAULT_TARGET,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[PairedComparison]:
    """Paired bootstrap comparisons for a saved result file.

    With ``baseline_result_path`` the target rows come from ``result_path``
    and baseline rows from the second file (any strategy named like the
    target there is ignored).
    """
    result = _load_result(result_path)
    aggregate = dict(result.get("aggregate") or {})
    if not aggregate.get(target):
        raise ValueError(
            f"'{target}' not found in {result_path}; "
            f"strategies: {', '.join(sorted(aggregate)) or '(none)'}"
        )
    config = result.get("config", {})
    metric_k = config.get("metric_k", 5)

    if baseline_result_path:
        baseline_result = _load_result(baseline_result_path)
        baseline_config = baseline_result.get("config", {})
        for key in ("dataset", "index_mode", "metric_k"):
            if baseline_config.get(key) != config.get(key):
                raise ValueError(
                    f"result files differ on {key}: {config.get(key)!r} vs "
                    f"{baseline_config.get(key)!r} - rows are not comparable"
                )
        aggregate = {target: aggregate[target]} | {
            name: rows
            for name, rows in (baseline_result.get("aggregate") or {}).items()
            if name != target
        }

    return compare_to_baselines(
        aggregate,
        target=target,
        metric_keys=[f"hr@{metric_k}", "mrr"],
        n_resamples=n_resamples,
        seed=seed,
    )
