"""Tests for dual-seed rank interleave (phantom ∪ clean)."""

import numpy as np

from src.seed_retrieval import dual_seed_indices, interleave_ranked_indices, ranked_indices


def test_ranked_indices_descending():
    sims = np.array([0.1, 0.9, 0.2, 0.3])
    assert list(ranked_indices(sims, 2)) == [1, 3]


def test_interleave_skips_duplicates_and_keeps_primary_lead():
    primary = np.array([1, 3, 2])
    secondary = np.array([1, 0, 2])
    assert list(interleave_ranked_indices(primary, secondary)) == [1, 0, 3, 2]


def test_dual_seed_indices_unions_top_k_from_each_space():
    # Primary (phantom) prefers index 1, then 3.
    # Clean prefers index 0, then 2 — the distinctive fact FW would retrieve.
    phantom = np.array([0.1, 0.9, 0.2, 0.3])
    clean = np.array([0.8, 0.1, 0.7, 0.0])
    got = dual_seed_indices(phantom, clean, k=2)
    assert list(got) == [1, 0, 3, 2]


def test_dual_seed_indices_without_secondary_is_ordinary_topk():
    phantom = np.array([0.1, 0.9, 0.2, 0.3])
    assert list(dual_seed_indices(phantom, None, k=2)) == [1, 3]
