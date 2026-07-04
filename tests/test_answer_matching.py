"""Contract tests for answer matching.

Chunk-level ``contains_answer`` must be (near-)verbatim: exact normalized
containment, plus a contiguous-token-window fallback for answer sentences
truncated at chunk boundaries. Set-based token coverage was removed from the
chunk level after it produced false positives on unrelated chunks via
function words alone (regression captured below).
"""

from src.answer_matching import (
    answer_match_score,
    contains_answer,
    find_answer_sentence_idx,
)

UNRELATED_CHUNK = """The city was founded in the early period and grew rapidly during the
industrial era. Many buildings in the old town were constructed near the river,
and the local government began a program of restoration. Over the years the
population increased, and the village that once stood at the site became part
of the larger metropolitan area. The museum of local history contains artifacts
from the region, and the university near the central square attracts students."""


def test_unrelated_chunk_with_shared_function_words_is_not_a_hit():
    # Regression: these answers previously matched via set-based token
    # coverage (scores 0.70-0.89) even though the chunk is about a city.
    answers = [
        "The battle began in 1815 near the village of Waterloo",
        "The treaty was signed in the city by the local government",
        "Construction of the museum began during the industrial era",
    ]
    for answer in answers:
        assert not contains_answer(UNRELATED_CHUNK, answer)


def test_verbatim_containment_is_a_hit():
    answer = "the local government began a program of restoration"
    assert contains_answer(UNRELATED_CHUNK, answer)


def test_containment_respects_token_boundaries():
    # "the cat" must not match inside "the catalog".
    assert not contains_answer("Look in the catalog now.", "the cat")


def test_truncated_answer_sentence_is_still_a_hit():
    answer = "The reaction requires a catalyst and proceeds at low temperature in the dark"
    # Token splitter clipped the last token of the sentence (12 of 13 kept).
    chunk = "Some context. The reaction requires a catalyst and proceeds at low temperature in the"
    assert contains_answer(chunk, answer)


def test_scattered_answer_tokens_are_not_a_hit():
    answer = "The reaction requires a catalyst and proceeds at low temperature in the dark"
    scattered = (
        "The dark room stores equipment. A catalyst is expensive. The reaction "
        "chamber requires cleaning. Temperature control proceeds slowly at low cost."
    )
    assert not contains_answer(scattered, answer)


def test_sentence_level_resolution_tolerates_paraphrase():
    sentences = [
        "A distractor sentence about alpha beta.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    paraphrased = "quick brown fox jumped over lazy dog"

    assert find_answer_sentence_idx(sentences, paraphrased) == 1
    assert answer_match_score(sentences[1], paraphrased) >= 0.70
    # The same paraphrase is not a chunk-level hit.
    assert not contains_answer(sentences[1], paraphrased)
