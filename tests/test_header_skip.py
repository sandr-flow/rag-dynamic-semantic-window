"""Section-header detection and expansion jump."""

import numpy as np
import pytest

from src.expansion_core import (
    DynamicExpansionCore,
    build_header_mask,
    is_section_header,
)


@pytest.mark.parametrize(
    "text",
    [
        "Methodology",
        "Introduction",
        "3. Experimental Setup",
        "3.1 Results",
        "RELATED WORK",
        "== Background ==",
        "A. Approach",
        "References",
        "Task Definition",
    ],
)
def test_is_section_header_accepts_headings(text):
    assert is_section_header(text)


@pytest.mark.parametrize(
    "text",
    [
        "The model was trained on five datasets.",
        "Figure 1: Accuracy on the test set",
        "A new method was proposed without a heading.",
        "Ok.",
        "This results in a total training dataset of 5000 posts.",
    ],
)
def test_is_section_header_rejects_prose(text):
    assert not is_section_header(text)


def _cliff_core(sentences, *, header_mask, max_expand=4):
    n = len(sentences)
    # Cliff on both sides of sentence 2; everything else is strongly adjacent.
    neighbor_sims = np.full(n - 1, 0.95, dtype=np.float32)
    neighbor_sims[1] = 0.15
    neighbor_sims[2] = 0.15
    sentence_sims = np.array([0.9, 0.4, 0.05, 0.35, 0.3], dtype=np.float32)[:n]
    return DynamicExpansionCore(
        neighbor_sims=neighbor_sims,
        sentence_sims=sentence_sims,
        top_k_indices=np.array([0], dtype=np.int32),
        threshold=0.8,
        skip_threshold=0.99,
        min_window=0,
        max_expand=max_expand,
        relevance_threshold_pct=1.0,
        merge_gap=0,
        target_clusters=1,
        adaptive_threshold_enabled=False,
        header_mask=header_mask,
    )


def test_header_jump_reaches_gold_across_section_cliff():
    sentences = [
        "The paraphrase of the answer sits here with plenty of words.",
        "Some bridging sentence about the same topic with more words.",
        "Methodology",
        "The gold answer sentence contains the distinctive fact we need.",
        "More methods discussion continues after the answer sentence here.",
    ]
    header_mask = build_header_mask(sentences)
    assert header_mask[2]

    clusters = _cliff_core(sentences, header_mask=header_mask).expand_and_retrieve()
    assert clusters
    assert clusters[0].start_idx == 0
    assert clusters[0].end_idx >= 3


def test_content_cliff_without_header_still_stops():
    sentences = [
        "The paraphrase of the answer sits here with plenty of words.",
        "Some bridging sentence about the same topic with more words.",
        "Unrelated methods prose that is not a heading at all here.",
        "The gold answer sentence contains the distinctive fact we need.",
        "More methods discussion continues after the answer sentence here.",
    ]
    clusters = _cliff_core(
        sentences, header_mask=build_header_mask(sentences)
    ).expand_and_retrieve()
    assert clusters
    assert clusters[0].end_idx == 1


def test_header_jump_does_not_consume_max_expand():
    sentences = [
        "The paraphrase of the answer sits here with plenty of words.",
        "Methodology",
        "The gold answer sentence contains the distinctive fact we need.",
        "Trailing methods discussion that should not be required here.",
    ]
    neighbor_sims = np.full(3, 0.1, dtype=np.float32)
    core = DynamicExpansionCore(
        neighbor_sims=neighbor_sims,
        sentence_sims=np.array([0.9, 0.05, 0.4, 0.2], dtype=np.float32),
        top_k_indices=np.array([0], dtype=np.int32),
        threshold=0.8,
        skip_threshold=0.99,
        min_window=0,
        max_expand=1,
        relevance_threshold_pct=1.0,
        merge_gap=0,
        target_clusters=1,
        adaptive_threshold_enabled=False,
        header_mask=build_header_mask(sentences),
    )
    clusters = core.expand_and_retrieve()
    assert clusters
    assert clusters[0].end_idx >= 2
