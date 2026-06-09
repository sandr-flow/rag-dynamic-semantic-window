"""Strategy registry and factory utilities."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from llama_index.core.schema import Document

from src.config import (
    DEFAULT_CODE_SPLITTER_CONFIG,
    DEFAULT_DYNAMIC_SEMANTIC_CONFIG,
    DEFAULT_EXPANSION_CONFIG,
    DEFAULT_FIXED_WINDOW_CONFIG,
    DEFAULT_HTML_SPLITTER_CONFIG,
    DEFAULT_JSON_SPLITTER_CONFIG,
    DEFAULT_MARKDOWN_SPLITTER_CONFIG,
    DEFAULT_NAIVE_CHUNKING_CONFIG,
    DEFAULT_SEMANTIC_SPLITTER_CONFIG,
    DEFAULT_TOKEN_TEXT_SPLITTER_CONFIG,
)
from src.strategies import (
    BaseStrategy,
    CodeSplitterStrategy,
    DynamicSemanticStrategy,
    FixedWindowStrategy,
    HTMLSplitterStrategy,
    JSONSplitterStrategy,
    MarkdownSplitterStrategy,
    NaiveChunkingStrategy,
    SemanticSplitterStrategy,
    TokenTextSplitterStrategy,
)

DEFAULT_STRATEGY_IDS = [
    "naive",
    "fixed_window",
    "token_text",
    "semantic_splitter",
    "dynamic_semantic",
]

ALL_STRATEGY_IDS = [
    *DEFAULT_STRATEGY_IDS,
    "markdown",
    "html",
    "json",
    "code",
]

STRATEGY_DESCRIPTIONS = {
    "naive": "SentenceSplitter baseline for plain text.",
    "fixed_window": "SentenceWindowNodeParser baseline with fixed neighbor context.",
    "token_text": "TokenTextSplitter baseline for token-bounded chunks.",
    "semantic_splitter": "SemanticSplitterNodeParser baseline using embedding breakpoints.",
    "dynamic_semantic": "Custom dynamic semantic window strategy.",
    "markdown": "MarkdownNodeParser for markdown documents.",
    "html": "HTMLNodeParser for HTML documents.",
    "json": "JSONNodeParser for structured JSON payloads.",
    "code": "CodeSplitter for source code documents.",
}

STRATEGY_OVERRIDE_KEYS = {
    "naive": ["chunk_size", "chunk_overlap"],
    "fixed_window": ["window_size"],
    "token_text": ["chunk_size", "chunk_overlap"],
    "semantic_splitter": ["buffer_size", "breakpoint_percentile_threshold"],
    "dynamic_semantic": [
        "phantom_window",
        "prefetch_multiplier",
        "threshold",
        "skip_threshold",
        "relevance_threshold_pct",
        "max_expand",
        "min_window",
        "min_chunk_length",
        "target_clusters",
        "merge_gap",
    ],
    "markdown": ["header_path_separator"],
    "html": ["tags"],
    "json": ["include_metadata"],
    "code": ["language", "chunk_lines", "chunk_lines_overlap", "max_chars", "count_mode", "max_tokens"],
}


ALIASES = {
    "naive": "naive",
    "naive_chunking": "naive",
    "sentence": "naive",
    "sentence_splitter": "naive",
    "fixed": "fixed_window",
    "fixed_window": "fixed_window",
    "sentence_window": "fixed_window",
    "token": "token_text",
    "token_text": "token_text",
    "token_text_splitter": "token_text",
    "semantic": "semantic_splitter",
    "semantic_splitter": "semantic_splitter",
    "dynamic": "dynamic_semantic",
    "dynamic_semantic": "dynamic_semantic",
    "markdown": "markdown",
    "markdown_splitter": "markdown",
    "markdown_node_parser": "markdown",
    "html": "html",
    "html_splitter": "html",
    "html_node_parser": "html",
    "json": "json",
    "json_splitter": "json",
    "json_node_parser": "json",
    "code": "code",
    "code_splitter": "code",
}


def normalize_strategy_id(strategy_id: str) -> str:
    normalized = strategy_id.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in ALIASES:
        choices = ", ".join(sorted(ALIASES))
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {choices}")
    return ALIASES[normalized]


def parse_strategy_ids(value: str | None) -> list[str]:
    """Parse comma-separated strategy ids."""
    if not value or value.lower() == "default":
        return list(DEFAULT_STRATEGY_IDS)
    if value.lower() == "all":
        return list(ALL_STRATEGY_IDS)
    return [normalize_strategy_id(item) for item in value.split(",") if item.strip()]


def available_strategy_ids() -> list[str]:
    return list(ALL_STRATEGY_IDS)


def strategy_catalog() -> list[dict[str, Any]]:
    """Return strategy metadata for CLI inspection and docs generation."""
    return [
        {
            "id": strategy_id,
            "default": strategy_id in DEFAULT_STRATEGY_IDS,
            "description": STRATEGY_DESCRIPTIONS[strategy_id],
            "override_keys": STRATEGY_OVERRIDE_KEYS[strategy_id],
            "default_params": default_strategy_params(strategy_id),
        }
        for strategy_id in ALL_STRATEGY_IDS
    ]


def default_strategy_params(strategy_id: str) -> dict[str, Any]:
    """Return user-tunable default params for one strategy."""
    normalized = normalize_strategy_id(strategy_id)
    defaults = _strategy_default_values(normalized)
    return {
        key: _jsonable_value(defaults[key])
        for key in STRATEGY_OVERRIDE_KEYS[normalized]
        if key in defaults
    }


def resolve_strategy_overrides(strategy_id: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Return flat overrides for one strategy.

    Supports both legacy flat params:
        {"threshold": 0.9, "max_expand": 4}

    and grouped params:
        {"global": {"chunk_size": 256}, "dynamic_semantic": {"threshold": 0.9}}
    """
    if not overrides:
        return {}

    normalized = normalize_strategy_id(strategy_id)
    has_grouped_values = any(
        _is_grouped_override_key(key) and isinstance(value, dict)
        for key, value in overrides.items()
    )
    if not has_grouped_values:
        return overrides if normalized == "dynamic_semantic" else {}

    resolved = {}
    for common_key in ("global", "default"):
        if isinstance(overrides.get(common_key), dict):
            resolved.update(overrides[common_key])

    for key, value in overrides.items():
        if not isinstance(value, dict):
            continue
        try:
            key_strategy = normalize_strategy_id(key)
        except ValueError:
            continue
        if key_strategy == normalized:
            resolved.update(value)

    return resolved


def validate_strategy_overrides_for_ids(
    strategy_ids: list[str],
    overrides: dict[str, Any] | None,
) -> list[str]:
    """Return validation errors for strategy override keys."""
    if not overrides:
        return []

    normalized_ids = [normalize_strategy_id(strategy_id) for strategy_id in strategy_ids]
    errors = []
    for strategy_id in normalized_ids:
        resolved = resolve_strategy_overrides(strategy_id, overrides)
        allowed = set(STRATEGY_OVERRIDE_KEYS[strategy_id])
        unknown_keys = sorted(set(resolved) - allowed)
        if unknown_keys:
            errors.append(
                f"{strategy_id} received unsupported override keys: {', '.join(unknown_keys)}"
            )

    if _looks_grouped(overrides):
        for key, value in overrides.items():
            if key in {"global", "default"}:
                if not isinstance(value, dict):
                    errors.append(f"{key} override must be an object")
                continue
            if not isinstance(value, dict):
                errors.append(f"grouped override for {key} must be an object")
                continue
            try:
                key_strategy = normalize_strategy_id(key)
            except ValueError:
                errors.append(f"unknown strategy override group: {key}")
                continue
            if key_strategy not in normalized_ids:
                errors.append(f"override group is not used by selected strategies: {key}")

    return errors


def _is_grouped_override_key(key: str) -> bool:
    if key in {"global", "default"}:
        return True
    try:
        normalize_strategy_id(key)
    except ValueError:
        return False
    return True


def _looks_grouped(overrides: dict[str, Any]) -> bool:
    return any(
        key in {"global", "default"} or isinstance(value, dict)
        for key, value in overrides.items()
    )


def create_strategy(
    strategy_id: str,
    documents: list[Document],
    top_k: int,
    seed_rejection_log_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> BaseStrategy:
    """Create a retrieval strategy by id."""
    strategy_id = normalize_strategy_id(strategy_id)
    overrides = overrides or {}

    if strategy_id == "naive":
        config = _replace(DEFAULT_NAIVE_CHUNKING_CONFIG, overrides, {"chunk_size", "chunk_overlap"})
        return NaiveChunkingStrategy(documents, top_k=top_k, config=config)

    if strategy_id == "fixed_window":
        config = _replace(DEFAULT_FIXED_WINDOW_CONFIG, overrides, {"window_size"})
        return FixedWindowStrategy(documents, top_k=top_k, config=config)

    if strategy_id == "token_text":
        config = _replace(DEFAULT_TOKEN_TEXT_SPLITTER_CONFIG, overrides, {"chunk_size", "chunk_overlap"})
        return TokenTextSplitterStrategy(documents, top_k=top_k, config=config)

    if strategy_id == "semantic_splitter":
        config = _replace(
            DEFAULT_SEMANTIC_SPLITTER_CONFIG,
            overrides,
            {"buffer_size", "breakpoint_percentile_threshold"},
        )
        return SemanticSplitterStrategy(documents, top_k=top_k, config=config)

    if strategy_id == "markdown":
        config = _replace(
            DEFAULT_MARKDOWN_SPLITTER_CONFIG,
            overrides,
            {"header_path_separator"},
        )
        return MarkdownSplitterStrategy(documents, top_k=top_k, config=config)

    if strategy_id == "html":
        config = _replace(DEFAULT_HTML_SPLITTER_CONFIG, overrides, {"tags"})
        return HTMLSplitterStrategy(documents, top_k=top_k, config=config)

    if strategy_id == "json":
        config = _replace(DEFAULT_JSON_SPLITTER_CONFIG, overrides, {"include_metadata"})
        return JSONSplitterStrategy(documents, top_k=top_k, config=config)

    if strategy_id == "code":
        config = _replace(
            DEFAULT_CODE_SPLITTER_CONFIG,
            overrides,
            {
                "language",
                "chunk_lines",
                "chunk_lines_overlap",
                "max_chars",
                "count_mode",
                "max_tokens",
            },
        )
        return CodeSplitterStrategy(documents, top_k=top_k, config=config)

    if strategy_id == "dynamic_semantic":
        dynamic_config = _replace(
            DEFAULT_DYNAMIC_SEMANTIC_CONFIG,
            overrides,
            {"phantom_window", "prefetch_multiplier"},
        )
        expansion_config = _replace(
            DEFAULT_EXPANSION_CONFIG,
            overrides,
            {
                "threshold",
                "skip_threshold",
                "relevance_threshold_pct",
                "max_expand",
                "min_window",
                "min_chunk_length",
                "target_clusters",
                "merge_gap",
            },
        )
        return DynamicSemanticStrategy(
            documents,
            top_k=top_k,
            seed_rejection_log_path=seed_rejection_log_path,
            dynamic_config=dynamic_config,
            expansion_config=expansion_config,
        )

    raise ValueError(f"Unsupported strategy: {strategy_id}")


def _replace(config, overrides: dict[str, Any], allowed_keys: set[str]):
    values = {key: value for key, value in overrides.items() if key in allowed_keys}
    return replace(config, **values) if values else config


def _strategy_default_values(strategy_id: str) -> dict[str, Any]:
    if strategy_id == "naive":
        return asdict(DEFAULT_NAIVE_CHUNKING_CONFIG)
    if strategy_id == "fixed_window":
        return asdict(DEFAULT_FIXED_WINDOW_CONFIG)
    if strategy_id == "token_text":
        return asdict(DEFAULT_TOKEN_TEXT_SPLITTER_CONFIG)
    if strategy_id == "semantic_splitter":
        return asdict(DEFAULT_SEMANTIC_SPLITTER_CONFIG)
    if strategy_id == "markdown":
        return asdict(DEFAULT_MARKDOWN_SPLITTER_CONFIG)
    if strategy_id == "html":
        return asdict(DEFAULT_HTML_SPLITTER_CONFIG)
    if strategy_id == "json":
        return asdict(DEFAULT_JSON_SPLITTER_CONFIG)
    if strategy_id == "code":
        return asdict(DEFAULT_CODE_SPLITTER_CONFIG)
    if strategy_id == "dynamic_semantic":
        return {
            **asdict(DEFAULT_DYNAMIC_SEMANTIC_CONFIG),
            **asdict(DEFAULT_EXPANSION_CONFIG),
        }
    raise ValueError(f"Unsupported strategy: {strategy_id}")


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    return value
