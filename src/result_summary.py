"""Summarize benchmark result JSON files into tabular records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def summarize_benchmark_file(path: str | Path) -> list[dict[str, Any]]:
    """Return one summary row per strategy for a benchmark JSON file."""
    result_path = Path(path)
    with open(result_path, encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config") or {}
    embedding = payload.get("embedding") or {}
    llm = payload.get("llm") or {}
    strategy_overrides = payload.get("strategy_overrides") or {}
    effective_strategy_overrides = payload.get("effective_strategy_overrides") or {}

    rows = []
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict):
        for strategy, metric_list in aggregate.items():
            if not isinstance(metric_list, list) or not metric_list:
                continue
            metric_summary = _average_metric_list(metric_list)
            rows.append(
                _base_row(
                    result_path,
                    config,
                    embedding,
                    llm,
                    strategy,
                    _overrides_for_strategy(strategy, strategy_overrides, effective_strategy_overrides),
                    metric_summary,
                )
            )
        return rows

    aggregate_metrics = payload.get("aggregate_metrics")
    if isinstance(aggregate_metrics, dict):
        for strategy, metrics in aggregate_metrics.items():
            metric_summary = _normalize_prefixed_metrics(metrics)
            rows.append(
                _base_row(
                    result_path,
                    config,
                    embedding,
                    llm,
                    strategy,
                    _overrides_for_strategy(strategy, strategy_overrides, effective_strategy_overrides),
                    metric_summary,
                )
            )

    return rows


def summarize_files(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(summarize_benchmark_file(path))
    return rows


def build_leaderboard(
    rows: list[dict[str, Any]],
    quality_metric: str | None = None,
    group_by: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Rank summary rows by quality descending and token cost ascending."""
    group_keys = list(group_by or [])
    ranked_rows = []
    for row in rows:
        metric_name = quality_metric or _default_quality_metric(row)
        metric_value = _to_float(row.get(metric_name))
        ranked = dict(row)
        ranked["rank_metric"] = metric_name
        ranked["rank_score"] = metric_value
        if group_keys:
            ranked["rank_group"] = _rank_group_value(ranked, group_keys)
        ranked_rows.append(ranked)

    ranked_rows.sort(
        key=lambda row: (
            *(str(row.get(key) or "") for key in group_keys),
            -_sort_number(row.get("rank_score"), missing_default=-1.0),
            _sort_number(row.get("avg_tokens"), missing_default=float("inf")),
            str(row.get("strategy") or ""),
        )
    )
    current_group = None
    rank = 0
    for global_index, row in enumerate(ranked_rows, start=1):
        group_value = tuple(row.get(key) for key in group_keys)
        if group_keys:
            if group_value != current_group:
                current_group = group_value
                rank = 1
            else:
                rank += 1
        else:
            rank = global_index
        row["rank"] = rank
    return ranked_rows


def write_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write rows to CSV with stable sorted columns."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_leaderboard_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write ranked rows to CSV with leaderboard columns first."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _leaderboard_fieldnames(rows)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(
    path: Path,
    config: dict[str, Any],
    embedding: dict[str, Any],
    llm: dict[str, Any],
    strategy: str,
    strategy_overrides: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "result_path": str(path),
        "result_file": path.name,
        "dataset_name": config.get("dataset_name"),
        "source": config.get("source"),
        "split": config.get("split"),
        "num_articles": config.get("actual_num_articles", config.get("num_articles")),
        "num_questions": config.get("actual_num_questions", config.get("num_questions")),
        "requested_num_articles": config.get("requested_num_articles", config.get("num_articles")),
        "questions_per_article": config.get("questions_per_article", config.get("num_questions")),
        "top_k": config.get("top_k"),
        "metric_k": payload_metric_k(config),
        "strategies": config.get("strategies"),
        "embedding_provider": embedding.get("provider"),
        "embedding_model": embedding.get("model"),
        "llm_provider": llm.get("provider"),
        "llm_model": llm.get("model"),
        "llm_used_for_qa_generation": llm.get("used_for_qa_generation"),
        "strategy": strategy,
        "strategy_overrides": json.dumps(strategy_overrides, sort_keys=True),
    }
    row.update(metrics)
    return row


def _average_metric_list(metric_list: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for metrics in metric_list for key, value in metrics.items() if _is_number(value)})
    return {
        f"avg_{key}": mean(float(metrics[key]) for metrics in metric_list if _is_number(metrics.get(key)))
        for key in keys
    }


def _normalize_prefixed_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key if key.startswith("avg_") else f"avg_{key}": value
        for key, value in metrics.items()
        if _is_number(value)
    }


def _overrides_for_strategy(
    strategy: str,
    strategy_overrides: dict[str, Any],
    effective_strategy_overrides: dict[str, Any],
) -> dict[str, Any]:
    if strategy in effective_strategy_overrides:
        return effective_strategy_overrides[strategy] or {}
    return strategy_overrides


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _to_float(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    return None


def _sort_number(value: Any, missing_default: float) -> float:
    parsed = _to_float(value)
    return parsed if parsed is not None else missing_default


def _rank_group_value(row: dict[str, Any], group_by: list[str]) -> str:
    return " | ".join(f"{key}={row.get(key) or ''}" for key in group_by)


def _default_quality_metric(row: dict[str, Any]) -> str:
    metric_k = payload_metric_k(row)
    candidates = []
    if metric_k is not None:
        candidates.extend([
            f"avg_ndcg@{metric_k}",
            f"avg_hr@{metric_k}",
            f"avg_recall@{metric_k}",
            f"avg_precision@{metric_k}",
        ])
    candidates.extend(["avg_ndcg@5", "avg_hr@5", "avg_mrr"])
    for candidate in candidates:
        if _is_number(row.get(candidate)):
            return candidate
    return "avg_mrr"


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "result_file",
        "dataset_name",
        "source",
        "split",
        "embedding_provider",
        "embedding_model",
        "llm_provider",
        "llm_model",
        "llm_used_for_qa_generation",
        "strategy",
        "top_k",
        "metric_k",
        "num_articles",
        "num_questions",
        "requested_num_articles",
        "questions_per_article",
        "avg_tokens",
        "avg_hr@5",
        "avg_mrr",
        "avg_precision@5",
        "avg_recall@5",
        "avg_ndcg@5",
        "result_path",
        "strategies",
        "strategy_overrides",
    ]
    present = {key for row in rows for key in row.keys()}
    return [key for key in preferred if key in present] + sorted(present - set(preferred))


def _leaderboard_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "rank",
        "rank_group",
        "rank_metric",
        "rank_score",
        "strategy",
        "dataset_name",
        "avg_tokens",
        "source",
        "embedding_provider",
        "embedding_model",
        "llm_provider",
        "llm_model",
        "top_k",
        "metric_k",
        "result_file",
        "result_path",
    ]
    present = {key for row in rows for key in row.keys()}
    return [key for key in preferred if key in present] + sorted(present - set(preferred))


def payload_metric_k(config: dict[str, Any]) -> Any:
    """Return metric cutoff, defaulting to top_k for older result files."""
    return config.get("metric_k") or config.get("top_k")
