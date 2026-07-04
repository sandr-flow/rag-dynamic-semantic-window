"""Tests for the shared token counting function (improvement plan step 1.1)."""

import warnings

import pytest

from src import tokens
from src.tokens import count_tokens


def test_empty_text_is_zero():
    assert count_tokens("") == 0


def test_matches_tiktoken_exactly():
    tiktoken = pytest.importorskip("tiktoken")
    encoder = tiktoken.get_encoding(tokens.TOKEN_ENCODING)
    text = "Dynamic semantic windows retrieve compact context, e.g. Fig. 3 shows 2.5x."
    assert count_tokens(text) == len(encoder.encode(text))


def test_special_token_strings_are_plain_text():
    # Corpus text may contain special-token markers; they must not raise.
    assert count_tokens("prefix <|endoftext|> suffix") > 0


def test_fallback_when_tiktoken_unavailable(monkeypatch):
    def broken_encoder():
        raise ImportError("no tiktoken")

    monkeypatch.setattr(tokens, "_token_encoder", broken_encoder)
    text = "x" * 40
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert count_tokens(text) == 10  # chars / 4
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
