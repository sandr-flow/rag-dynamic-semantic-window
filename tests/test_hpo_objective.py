"""Tests for the HPO scoring policy (dynamic token tie-breaker)."""

import pytest

from src.hpo_config import (
    DEFAULT_FLOAT_STEP,
    ObjectivePolicy,
    compute_objective_score,
    load_hpo_settings,
    quantize_float,
    quantize_params,
    search_space_for,
)


def _score(policy, *, hr, tokens, mrr=0.5, n=200):
    return compute_objective_score(
        avg_hr=hr,
        avg_mrr=mrr,
        avg_tokens=tokens,
        num_valid_questions=n,
        policy=policy,
    )[0]


def _yaml_policy(**kwargs):
    values = dict(
        mrr_weight=0.4,
        token_tiebreak_fraction=0.05,
        token_overage_power=2.0,
        token_overage_scale=10.0,
    )
    values.update(kwargs)
    return ObjectivePolicy(**values)


def test_token_bonus_weight_is_a_strict_sub_hr_tiebreaker():
    policy = ObjectivePolicy()
    # For any question count, the largest possible token bonus is strictly
    # smaller than the value of a single caught question.
    for n in (1, 10, 200, 5000):
        one_question = policy.hr_weight / n
        assert policy.token_bonus_weight_for(n) < one_question
        assert policy.token_bonus_weight_for(n) == pytest.approx(0.5 * one_question)


def test_fewer_tokens_win_at_equal_hit_rate():
    policy = ObjectivePolicy()
    cheap = _score(policy, hr=0.9, tokens=400)
    expensive = _score(policy, hr=0.9, tokens=900)
    assert cheap > expensive


def test_one_more_question_beats_any_token_saving():
    policy = ObjectivePolicy()
    # Worst case for the higher-HR config: it sits at the token reference (no bonus)
    # and has the lower MRR; the cheaper config catches one fewer question but
    # gets the maximum token bonus. HR must still win.
    n = 200
    better_hr = _score(policy, hr=0.90 + 1 / n, tokens=policy.soft_token_limit, mrr=0.0, n=n)
    cheaper = _score(policy, hr=0.90, tokens=0, mrr=1.0, n=n)
    assert better_hr > cheaper


def test_modest_overage_still_loses_to_one_extra_hit():
    """~1.25x the reference is cheaper than catching one more question."""
    policy = _yaml_policy()
    n = 200
    modest = _score(policy, hr=0.90 + 1 / n, tokens=1500, mrr=0.5, n=n)
    at_limit = _score(policy, hr=0.90, tokens=policy.soft_token_limit, mrr=0.5, n=n)
    assert modest > at_limit


def test_3k_tokens_lose_to_one_fewer_hit():
    """Quadratic overage: 2.5x the reference costs several questions."""
    policy = _yaml_policy()
    n = 200
    bloated = _score(policy, hr=0.90 + 1 / n, tokens=3000, mrr=0.5, n=n)
    lean = _score(policy, hr=0.90, tokens=policy.soft_token_limit, mrr=0.5, n=n)
    assert lean > bloated


def test_fat_window_overage_loses_to_one_fewer_hit():
    """~1.5x the reference already costs more than catching one extra question."""
    policy = _yaml_policy()
    n = 200
    fat = _score(policy, hr=0.90 + 1 / n, tokens=1800, mrr=0.5, n=n)
    lean = _score(policy, hr=0.90, tokens=policy.soft_token_limit, mrr=0.5, n=n)
    assert lean > fat


def test_overage_penalty_accelerates():
    """The same +600 tokens costs more at 2400→3000 than at 1200→1800."""
    policy = _yaml_policy()
    n = 200
    early = _score(policy, hr=0.9, tokens=1200, n=n) - _score(policy, hr=0.9, tokens=1800, n=n)
    late = _score(policy, hr=0.9, tokens=2400, n=n) - _score(policy, hr=0.9, tokens=3000, n=n)
    assert late > early > 0


def test_same_hr_fewer_tokens_win_on_both_sides_of_reference():
    policy = ObjectivePolicy()
    assert _score(policy, hr=0.9, tokens=400) > _score(policy, hr=0.9, tokens=900)
    assert _score(policy, hr=0.9, tokens=1300) > _score(policy, hr=0.9, tokens=1800)


def test_extreme_over_budget_score_is_finite_and_ordered():
    policy = ObjectivePolicy()
    slightly_over = _score(policy, hr=1.0, tokens=policy.soft_token_limit + 10)
    far_over = _score(policy, hr=1.0, tokens=policy.soft_token_limit + 50_000)
    in_budget = _score(policy, hr=1.0, tokens=policy.soft_token_limit)
    assert slightly_over < in_budget
    assert far_over < slightly_over
    assert far_over > policy.invalid_score


def test_token_savings_outrank_mrr():
    policy = ObjectivePolicy()
    # Same HR: a big token saving must beat a full-MRR advantage.
    saves_tokens = _score(policy, hr=0.9, tokens=0, mrr=0.0)
    better_mrr = _score(policy, hr=0.9, tokens=policy.soft_token_limit, mrr=1.0)
    assert saves_tokens > better_mrr


def test_fraction_zero_falls_back_to_static_weight():
    policy = ObjectivePolicy(token_tiebreak_fraction=0.0, token_bonus_weight=0.0)
    # No bonus at all: equal HR/MRR configs score identically regardless of tokens.
    assert _score(policy, hr=0.9, tokens=100) == _score(policy, hr=0.9, tokens=1100)
    assert policy.token_bonus_weight_for(200) == 0.0


def test_search_space_for_each_strategy():
    dynamic = search_space_for("dynamic_semantic")
    assert {"threshold", "min_window", "max_expand"} <= set(dynamic)

    naive = search_space_for("naive")
    assert naive["chunk_size"].low == 128
    assert naive["chunk_overlap"].high < naive["chunk_size"].low

    token = search_space_for("token_text")
    assert set(token) == set(naive)

    fixed = search_space_for("fixed_window")
    assert set(fixed) == {"window_size"}
    assert fixed["window_size"].low == 1

    semantic = search_space_for("semantic_splitter")
    assert "breakpoint_percentile_threshold" in semantic


def test_load_hpo_settings_uses_strategy_space():
    settings = load_hpo_settings(strategy="fixed_window")
    assert set(settings.search_space) == {"window_size"}


def test_min_window_2_hpo_config_pins_safety_net():
    settings = load_hpo_settings(path="configs/hpo_mrr_first_min_window_2.yaml")
    assert settings.search_space["min_window"].low == 2
    assert settings.search_space["min_window"].high == 2
    assert settings.search_space["max_expand"].low == 2
    assert settings.objective.mrr_weight == 0.4


def test_mrr_first_config_enables_quadratic_overage():
    settings = load_hpo_settings(path="configs/hpo_mrr_first.yaml")
    assert settings.objective.token_overage_power == 2.0
    assert settings.objective.token_overage_scale == 10.0
    assert settings.objective.mrr_weight == 0.4


def test_float_search_step_is_two_decimals():
    dynamic = search_space_for("dynamic_semantic")
    assert dynamic["threshold"].step == DEFAULT_FLOAT_STEP
    assert dynamic["skip_threshold"].step == DEFAULT_FLOAT_STEP
    assert dynamic["relevance_threshold_pct"].step == DEFAULT_FLOAT_STEP


def test_quantize_float_drops_binary_dust():
    assert quantize_float(0.8979999999999999, 0.01) == 0.90
    assert quantize_float(0.9650000000000001, 0.01) == 0.97
    assert quantize_float(0.42, 0.01) == 0.42
    assert quantize_float(0.8979999999999999, 0.001) == 0.898


def test_quantize_params_rounds_floats_only():
    rounded = quantize_params(
        {
            "threshold": 0.8979999999999999,
            "min_window": 3,
            "dual_seed": True,
        }
    )
    assert rounded["threshold"] == 0.90
    assert rounded["min_window"] == 3
    assert rounded["dual_seed"] is True
