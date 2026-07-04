"""Tests for paired bootstrap significance (src/significance.py, stand/analysis.py)."""

import numpy as np
import pytest

from src.significance import (
    compare_to_baselines,
    format_comparisons,
    paired_bootstrap_ci,
)


def test_identical_arrays_give_zero_delta_and_degenerate_ci():
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    delta, lo, hi = paired_bootstrap_ci(values, values)
    assert delta == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_constant_shift_is_significant():
    rng = np.random.default_rng(0)
    baseline = rng.random(100)
    target = baseline + 0.5
    delta, lo, hi = paired_bootstrap_ci(target, baseline)
    assert delta == pytest.approx(0.5)
    assert lo == pytest.approx(0.5, abs=1e-9)
    assert hi == pytest.approx(0.5, abs=1e-9)


def test_pure_noise_is_not_significant():
    rng = np.random.default_rng(7)
    target = rng.normal(size=300)
    baseline = rng.normal(size=300)  # independent, same mean
    _, lo, hi = paired_bootstrap_ci(target, baseline, seed=1)
    assert lo <= 0.0 <= hi


def test_same_seed_is_deterministic():
    rng = np.random.default_rng(3)
    a = rng.random(50)
    b = rng.random(50)
    assert paired_bootstrap_ci(a, b, seed=9) == paired_bootstrap_ci(a, b, seed=9)


def test_rejects_mismatched_or_empty_arrays():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 0.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])


def _rows(hr_values, mrr_values):
    return [
        {"hr@5": hr, "mrr": mrr}
        for hr, mrr in zip(hr_values, mrr_values, strict=True)
    ]


def test_compare_to_baselines_flags_and_pairing():
    n = 40
    aggregate = {
        "Dynamic Semantic": _rows([1.0] * n, [0.9] * n),
        "Weak": _rows([0.0] * n, [0.4] * n),
        "Equal": _rows([1.0] * n, [0.9] * n),
    }
    comparisons = compare_to_baselines(
        aggregate, target="Dynamic Semantic", metric_keys=["hr@5", "mrr"]
    )
    by_key = {(c.baseline, c.metric): c for c in comparisons}

    strong = by_key[("Weak", "hr@5")]
    assert strong.delta == pytest.approx(1.0)
    assert strong.significant
    assert by_key[("Weak", "mrr")].delta == pytest.approx(0.5)

    equal = by_key[("Equal", "hr@5")]
    assert equal.delta == 0.0
    assert not equal.significant


def test_compare_to_baselines_skips_unpaired_strategy():
    aggregate = {
        "Dynamic Semantic": _rows([1.0, 1.0, 0.0], [0.9, 0.9, 0.1]),
        "Short": _rows([1.0], [0.5]),
    }
    with pytest.warns(UserWarning, match="not paired"):
        comparisons = compare_to_baselines(
            aggregate, target="Dynamic Semantic", metric_keys=["hr@5"]
        )
    assert comparisons == []


def test_format_comparisons_marks_non_significant():
    aggregate = {
        "Dynamic Semantic": _rows([1.0, 1.0, 1.0, 1.0], [0.9, 0.8, 0.9, 0.8]),
        "Base": _rows([0.0, 0.0, 0.0, 0.0], [0.9, 0.8, 0.8, 0.9]),
    }
    comparisons = compare_to_baselines(
        aggregate, target="Dynamic Semantic", metric_keys=["hr@5", "mrr"]
    )
    lines = format_comparisons(comparisons)
    table = "\n".join(lines)
    assert "dhr@5" in table
    assert " ns" in table  # the mrr delta hovers around zero
    assert "Base" in table


def test_cross_file_significance_report(tmp_path):
    import json

    from stand.analysis import significance_report

    config = {"dataset": "d", "index_mode": "per_document", "metric_k": 5}
    target_file = tmp_path / "target.json"
    baseline_file = tmp_path / "baseline.json"
    target_file.write_text(
        json.dumps(
            {
                "config": config,
                "aggregate": {"Dynamic Semantic": _rows([1.0, 1.0, 0.0], [0.9, 0.9, 0.0])},
            }
        ),
        encoding="utf-8",
    )
    baseline_file.write_text(
        json.dumps(
            {
                "config": config,
                # A same-named strategy in the baseline file must be ignored
                "aggregate": {
                    "Dynamic Semantic": _rows([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                    "Naive": _rows([0.0, 1.0, 0.0], [0.1, 0.9, 0.1]),
                },
            }
        ),
        encoding="utf-8",
    )

    comparisons = significance_report(
        str(target_file), baseline_result_path=str(baseline_file)
    )
    assert {c.baseline for c in comparisons} == {"Naive"}

    # Config mismatch must be rejected
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps({"config": {**config, "index_mode": "shared"}, "aggregate": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index_mode"):
        significance_report(str(target_file), baseline_result_path=str(other))
