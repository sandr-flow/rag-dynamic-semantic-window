"""Tests for the HPO scoring policy (dynamic token tie-breaker)."""

import pytest

from src.hpo_config import ObjectivePolicy, compute_objective_score


def _score(policy, *, hr, tokens, mrr=0.5, n=200):
    return compute_objective_score(
        avg_hr=hr,
        avg_mrr=mrr,
        avg_tokens=tokens,
        num_valid_questions=n,
        policy=policy,
    )[0]


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
    # Worst case for the higher-HR config: it sits at the token wall (no bonus)
    # and has the lower MRR; the cheaper config catches one fewer question but
    # gets the maximum token bonus. HR must still win.
    n = 200
    better_hr = _score(policy, hr=0.90 + 1 / n, tokens=policy.soft_token_limit, mrr=0.0, n=n)
    cheaper = _score(policy, hr=0.90, tokens=0, mrr=1.0, n=n)
    assert better_hr > cheaper


def test_token_savings_outrank_mrr():
    policy = ObjectivePolicy()
    # Same HR: a big token saving must beat a full-MRR advantage.
    saves_tokens = _score(policy, hr=0.9, tokens=0, mrr=0.0)
    better_mrr = _score(policy, hr=0.9, tokens=policy.soft_token_limit, mrr=1.0)
    assert saves_tokens > better_mrr


def test_over_budget_is_worse_than_any_in_budget_and_ordered():
    policy = ObjectivePolicy()
    in_budget = _score(policy, hr=0.0, tokens=policy.soft_token_limit)
    slightly_over = _score(policy, hr=1.0, tokens=policy.soft_token_limit + 10)
    far_over = _score(policy, hr=1.0, tokens=policy.soft_token_limit + 500)
    assert slightly_over < in_budget
    assert far_over < slightly_over


def test_fraction_zero_falls_back_to_static_weight():
    policy = ObjectivePolicy(token_tiebreak_fraction=0.0, token_bonus_weight=0.0)
    # No bonus at all: equal HR/MRR configs score identically regardless of tokens.
    assert _score(policy, hr=0.9, tokens=100) == _score(policy, hr=0.9, tokens=1100)
    assert policy.token_bonus_weight_for(200) == 0.0
