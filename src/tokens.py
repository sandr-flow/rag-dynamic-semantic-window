"""Token counting - single source of truth for the whole stand.

Used by the benchmark path (stand/runner.py) and the HPO surrogate
(src/expansion_core.py::evaluate_retrieval), so both report tokens on the
same scale. Kept dependency-light: importable without LlamaIndex.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

# Encoding used for all token accounting in the project (benchmark, HPO,
# token budgets). cl100k_base is the GPT-4/embedding-era BPE - a reasonable
# universal proxy for "how much context does this text cost".
TOKEN_ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _token_encoder():
    import tiktoken

    return tiktoken.get_encoding(TOKEN_ENCODING)


def count_tokens(text: str) -> int:
    """
    Count tokens in text with tiktoken.

    Falls back to the legacy chars/4 estimate (with a one-time warning) when
    tiktoken is unavailable, so smoke tests keep working in bare environments.
    """
    if not text:
        return 0
    try:
        encoder = _token_encoder()
    except Exception:
        warnings.warn(
            "tiktoken unavailable - falling back to chars/4 token estimate; "
            "token metrics will not match tiktoken-based runs",
            RuntimeWarning,
            stacklevel=2,
        )
        return len(text) // 4
    # disallowed_special=() treats special-token strings in corpus text
    # (e.g. "<|endoftext|>") as plain text instead of raising.
    return len(encoder.encode(text, disallowed_special=()))
