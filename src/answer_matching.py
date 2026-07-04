"""Shared answer matching helpers for metrics and corpus preparation.

Two deliberately different criteria live here:

- ``contains_answer`` (chunk-level, used by retrieval metrics): exact
  normalized containment, with a contiguous-token-window fallback that only
  tolerates truncation of the answer sentence at chunk boundaries. Set-based
  token coverage is deliberately NOT used here: on large chunks it produces
  false positives from function words alone.
- ``find_answer_sentence_idx`` (sentence-level, used to resolve the ground
  truth sentence when building HPO corpora): exact containment first, then a
  fuzzy best-match fallback (token coverage + string similarity) that can
  absorb light paraphrasing. Candidates are single sentences, so set-based
  coverage is acceptable in this context.

A chunk-level hit therefore requires (near-)verbatim presence of the answer
text, while ground-truth sentence resolution stays tolerant to paraphrased
answers.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_answer_text(text: str) -> str:
    """Normalize text for answer matching."""
    normalized = _PUNCT_RE.sub(" ", (text or "").casefold())
    return " ".join(normalized.split())


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_answer_text(text))


def answer_match_score(candidate_text: str, answer_text: str) -> float:
    """
    Sentence-level fuzzy containment score in [0, 1].

    Exact normalized substring matches score 1.0. The fallback uses answer-token
    coverage plus a string-similarity check for short, near-verbatim variants.

    Only meaningful for short candidates (single sentences): on large chunks
    set-based token coverage matches via function words alone — use
    ``contains_answer`` for chunk-level checks instead.
    """
    candidate_norm = normalize_answer_text(candidate_text)
    answer_norm = normalize_answer_text(answer_text)
    if not candidate_norm or not answer_norm:
        return 0.0
    if answer_norm in candidate_norm:
        return 1.0

    answer_tokens = _tokens(answer_text)
    if not answer_tokens:
        return 0.0
    candidate_tokens = set(_tokens(candidate_text))
    token_coverage = sum(1 for token in answer_tokens if token in candidate_tokens) / len(
        answer_tokens
    )

    char_similarity = SequenceMatcher(None, candidate_norm, answer_norm).ratio()
    return max(token_coverage, char_similarity)


def contains_answer(
    candidate_text: str,
    answer_text: str,
    *,
    min_coverage: float = 0.90,
) -> bool:
    """
    Return True when a chunk contains the expected answer (near-)verbatim.

    Exact normalized containment counts as a hit. Otherwise the chunk must
    contain at least ``min_coverage`` of the answer tokens as one contiguous
    run — this tolerates answer sentences truncated at chunk boundaries
    (token splitters) without admitting scattered-token false positives.
    """
    candidate_norm = normalize_answer_text(candidate_text)
    answer_norm = normalize_answer_text(answer_text)
    if not candidate_norm or not answer_norm:
        return False

    padded_candidate = f" {candidate_norm} "
    if f" {answer_norm} " in padded_candidate:
        return True

    answer_tokens = answer_norm.split()
    window = max(1, math.ceil(min_coverage * len(answer_tokens)))
    if window >= len(answer_tokens):
        # The full window equals the exact check that already failed.
        return False
    for start in range(len(answer_tokens) - window + 1):
        fragment = " ".join(answer_tokens[start : start + window])
        if f" {fragment} " in padded_candidate:
            return True
    return False


def find_answer_sentence_idx(
    sentences: list[str],
    answer_text: str,
    *,
    min_score: float = 0.70,
) -> int:
    """Find the best matching answer sentence index, or -1 if none qualifies."""
    answer_norm = normalize_answer_text(answer_text)
    if not answer_norm:
        return -1

    for i, sentence in enumerate(sentences):
        if answer_norm in normalize_answer_text(sentence):
            return i

    best_idx = -1
    best_score = 0.0
    for i, sentence in enumerate(sentences):
        score = answer_match_score(sentence, answer_text)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_score >= min_score else -1
