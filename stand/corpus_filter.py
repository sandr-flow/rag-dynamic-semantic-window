"""Shared corpus filters used by prep and benchmark runner."""

from __future__ import annotations

from typing import Any

from src.tokens import count_tokens

# Baseline parsers embed nltk-tokenized sentences as-is; giant blobs break
# cloud embedding APIs (OpenAI caps one input at 8192 tokens).
MAX_SENTENCE_TOKENS = 2000


def drop_unchunkable_items(
    items: list[dict[str, Any]], *, verbose: bool = False
) -> list[dict[str, Any]]:
    """Drop documents whose nltk sentence split yields oversized embedding inputs."""
    from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer

    split = split_by_sentence_tokenizer()
    kept = []
    for item in items:
        max_tokens = max((count_tokens(s) for s in split(item["text"])), default=0)
        if max_tokens > MAX_SENTENCE_TOKENS:
            if verbose:
                print(
                    f"  [skip-doc] '{item.get('title', '?')}': nltk sentence of "
                    f"{max_tokens} tokens exceeds {MAX_SENTENCE_TOKENS} (unchunkable)"
                )
            continue
        kept.append(item)
    return kept
