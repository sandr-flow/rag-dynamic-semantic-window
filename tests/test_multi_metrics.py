"""Tests for multi-answer retrieval metrics (extrahard)."""

from src.metrics import compute_multi_answer_metrics


def test_joint_hr_requires_both_answers():
    answers = ["Alpha beta.", "Delta epsilon."]
    both = ["chunk with Alpha beta.", "chunk with Delta epsilon."]
    one = ["chunk with Alpha beta.", "nothing relevant"]
    metrics_both = compute_multi_answer_metrics(both, answers, k=2)
    metrics_one = compute_multi_answer_metrics(one, answers, k=2)

    assert metrics_both["hr@2"] == 1.0
    assert metrics_one["hr@2"] == 0.0
    assert metrics_one["partial_hr@2"] == 1.0
    assert metrics_one["answer_recall@2"] == 0.5


def test_multi_mrr_averages_per_answer():
    answers = ["Alpha beta.", "Delta epsilon."]
    ranked = ["Alpha beta.", "irrelevant", "Delta epsilon."]
    metrics = compute_multi_answer_metrics(ranked, answers, k=3)
    assert metrics["mrr"] > 0.0
    assert metrics["mrr_min"] <= metrics["mrr"]
