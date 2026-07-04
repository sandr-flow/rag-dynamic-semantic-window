"""In-process benchmark runner — one combination per call.

This is the thin, non-interactive core that both the menu and CI use. No
argparse, no subprocess: ``run(config)`` does the work and returns the result.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from llama_index.core import Document

from src.metrics import compute_all_metrics
from src.strategy_registry import (
    STRATEGY_OVERRIDE_KEYS,
    create_strategy,
    normalize_strategy_id,
)
from src.tokens import count_tokens

from . import artifacts, paths
from .embeddings import prepare_embed_model, report_cache
from .runconfig import RunConfig


def _resolve_tuned(config: RunConfig) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return dynamic_semantic overrides and tuned manifest, if requested.

    Manual ``config.dynamic_overrides`` are applied on top of the
    default/tuned params so ablations can pin individual knobs.
    """
    params: dict[str, Any] = {}
    tuned = None
    if config.params == "tuned":
        tuned = artifacts.load_tuned(config.dataset, config.embedding)
        if tuned is None:
            raise ValueError(
                f"No tuned params for dataset '{config.dataset}' + embedding "
                f"'{config.embedding}'. Run: python -m stand tune ..."
            )
        params.update(tuned.get("params", {}))

    if config.dynamic_overrides:
        allowed = set(STRATEGY_OVERRIDE_KEYS["dynamic_semantic"])
        unknown = sorted(set(config.dynamic_overrides) - allowed)
        if unknown:
            raise ValueError(
                "Unsupported dynamic_overrides keys: "
                f"{', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}"
            )
        params.update(config.dynamic_overrides)

    return params, tuned


def _document_from_item(item: dict[str, Any], index: int) -> Document:
    metadata = {
        "source_doc": str(item.get("id", index)),
        "title": item.get("title", f"item_{index}"),
    }
    # Identification metadata only: node parsers propagate these exclusions,
    # so baseline strategies do not embed "source_doc: ...\ntitle: ..." text.
    return Document(
        text=item["text"],
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata.keys()),
        excluded_llm_metadata_keys=list(metadata.keys()),
    )


def _should_log_step(current: int, total: int) -> bool:
    """Log every step for small batches; ~10 milestones for larger ones."""
    if total <= 10:
        return True
    stride = max(1, total // 10)
    return current == 1 or current == total or current % stride == 0


def _build_strategies(
    strategy_ids: list[str],
    documents: list[Document],
    top_k: int,
    dynamic_overrides: dict[str, Any],
    failed: set[str],
    *,
    verbose: bool = False,
    log_prefix: str = "",
):
    """Instantiate every strategy over the same document set.

    A strategy that fails to build on this document set is skipped instead of
    crashing the whole run; its id is recorded in ``failed`` so callers can warn
    once and stop retrying it.
    """
    strategies = []
    for i, strategy_id in enumerate(strategy_ids, 1):
        normalized = normalize_strategy_id(strategy_id)
        if normalized in failed:
            continue
        if verbose:
            print(f"     {log_prefix}index {normalized} ({i}/{len(strategy_ids)})")
        overrides = dynamic_overrides if normalized == "dynamic_semantic" else {}
        try:
            strategies.append(
                create_strategy(normalized, documents, top_k=top_k, overrides=overrides)
            )
        except Exception as exc:  # incompatible parser for this document type
            failed.add(normalized)
            print(f"  [skip] strategy '{normalized}' failed to build: "
                  f"{type(exc).__name__}: {exc}")
    return strategies


def _evaluate(
    strategies,
    qa_pairs: list[dict],
    metric_k: int,
    *,
    verbose: bool = False,
    log_prefix: str = "",
) -> dict[str, list[dict]]:
    """Run every qa pair against every (already built) strategy."""
    results: dict[str, list[dict]] = {s.name: [] for s in strategies}
    total = len(qa_pairs)
    for qi, qa in enumerate(qa_pairs, 1):
        if verbose and _should_log_step(qi, total):
            print(f"     {log_prefix}eval {qi}/{total} questions")
        question = qa["question"]
        answer = qa.get("answer_sentence", qa.get("answer", ""))
        for strategy in strategies:
            nodes = strategy.retrieve(question)
            texts = [n.node.text for n in nodes]
            metrics = compute_all_metrics(texts, answer, k=metric_k)
            metrics["tokens"] = count_tokens(" ".join(texts))
            results[strategy.name].append(metrics)
    return results


def _merge(into: dict[str, list[dict]], extra: dict[str, list[dict]]) -> None:
    for name, metric_list in extra.items():
        into.setdefault(name, []).extend(metric_list)


def run(config: RunConfig, *, verbose: bool = True) -> dict[str, Any]:
    """Benchmark one combination and persist the result. Returns the result dict."""
    strategy_ids = [normalize_strategy_id(s) for s in config.strategies]
    metric_k = config.effective_metric_k
    dynamic_overrides, tuned_manifest = _resolve_tuned(config)

    embed_model, embedding_config = prepare_embed_model(config.embedding)

    items = artifacts.load_dataset_items(config.dataset, limit=config.limit)
    if not items:
        raise ValueError(f"Dataset '{config.dataset}' has no items with qa_pairs")

    if verbose:
        print("=" * 64)
        print("Dynamic Semantic Window Benchmark")
        print("=" * 64)
        print(f"Dataset:    {config.dataset} ({len(items)} docs)")
        print(f"Embedding:  {embedding_config.provider}/{embedding_config.model}")
        print(f"Strategies: {', '.join(strategy_ids)}")
        print(f"Index mode: {config.index_mode}")
        print(f"Params:     {config.params}")
        print(f"Top-K: {config.top_k}, Metric-K: {metric_k}")
        if dynamic_overrides:
            print(f"Dynamic params (tuned/overrides): {dynamic_overrides}")
        if tuned_manifest and tuned_manifest.get("metrics_val"):
            val = tuned_manifest["metrics_val"]
            print(
                "Tuned validation expectation: "
                f"HR={val.get('hit_rate', 0):.4f}, "
                f"MRR={val.get('mrr', 0):.4f}, "
                f"tokens={val.get('avg_tokens', 0):.0f}"
            )
        print("-" * 64)

    start = time.time()
    aggregate: dict[str, list[dict]] = {}
    failed: set[str] = set()

    if config.index_mode == "shared":
        documents = [_document_from_item(item, idx) for idx, item in enumerate(items)]
        all_qa = [qa for item in items for qa in item["qa_pairs"]]
        shared_prefix = f"[shared {len(documents)} docs, {len(all_qa)} q]"
        if verbose:
            print(f"  {shared_prefix} indexing...")
        shared_start = time.time()
        strategies = _build_strategies(
            strategy_ids,
            documents,
            config.top_k,
            dynamic_overrides,
            failed,
            verbose=verbose,
            log_prefix=f"{shared_prefix} ",
        )
        if verbose:
            print(f"  {shared_prefix} evaluating...")
        _merge(
            aggregate,
            _evaluate(
                strategies,
                all_qa,
                metric_k,
                verbose=verbose,
                log_prefix=f"{shared_prefix} ",
            ),
        )
        if verbose:
            print(f"  {shared_prefix} done ({time.time() - shared_start:.1f}s)")
    else:  # per_document
        n_docs = len(items)
        for doc_idx, item in enumerate(items, 1):
            n_q = len(item["qa_pairs"])
            doc_prefix = f"[{doc_idx}/{n_docs}]"
            if verbose:
                print(f"  {doc_prefix} {item['title']} ({n_q} q) - indexing...")
            doc_start = time.time()
            documents = [_document_from_item(item, doc_idx - 1)]
            strategies = _build_strategies(
                strategy_ids,
                documents,
                config.top_k,
                dynamic_overrides,
                failed,
                verbose=verbose,
                log_prefix=f"{doc_prefix} ",
            )
            if verbose:
                print(f"     {doc_prefix} evaluating...")
            _merge(
                aggregate,
                _evaluate(
                    strategies,
                    item["qa_pairs"],
                    metric_k,
                    verbose=verbose,
                    log_prefix=f"{doc_prefix} ",
                ),
            )
            if verbose:
                print(f"     {doc_prefix} done ({time.time() - doc_start:.1f}s)")

    if failed and not aggregate:
        raise ValueError(
            "No strategy could be built for this dataset. Skipped: " + ", ".join(sorted(failed))
        )

    report_cache(embed_model, verbose=verbose)

    total_questions = sum(len(item["qa_pairs"]) for item in items)
    summary = _summarize(aggregate, metric_k)
    if verbose:
        _print_table(summary, metric_k)
        print(f"\nTotal: {time.time() - start:.1f}s")

    result = {
        "config": _config_payload(config, embedding_config),
        "dataset": {"name": config.dataset, "docs": len(items), "questions": total_questions},
        "summary": summary,
        "aggregate": {name: [dict(m) for m in rows] for name, rows in aggregate.items()},
    }
    if tuned_manifest:
        result["tuned"] = {
            "created_at": tuned_manifest.get("created_at"),
            "metrics_train": tuned_manifest.get("metrics_train"),
            "metrics_val": tuned_manifest.get("metrics_val"),
            "tuning": tuned_manifest.get("tuning"),
        }
    _save_result(result, config)
    return result


def _summarize(aggregate: dict[str, list[dict]], metric_k: int) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in aggregate.items():
        if not metrics:
            continue
        rows.append(
            {
                "strategy": name,
                "tokens": float(np.mean([m["tokens"] for m in metrics])),
                f"hr@{metric_k}": float(np.mean([m[f"hr@{metric_k}"] for m in metrics])),
                "mrr": float(np.mean([m["mrr"] for m in metrics])),
                f"precision@{metric_k}": float(np.mean([m[f"precision@{metric_k}"] for m in metrics])),
                f"ndcg@{metric_k}": float(np.mean([m[f"ndcg@{metric_k}"] for m in metrics])),
            }
        )
    return rows


def _print_table(summary: list[dict[str, Any]], metric_k: int) -> None:
    if not summary:
        print("No results.")
        return
    suffix = f"@{metric_k}"
    print("-" * 85)
    print(
        f"{'Strategy':20} | {'Tokens':>7} | {f'HR{suffix}':>7} | "
        f"{'MRR':>7} | {f'P{suffix}':>7} | {f'NDCG{suffix}':>8}"
    )
    print("-" * 85)
    for row in summary:
        print(
            f"{row['strategy']:20} | {row['tokens']:7.1f} | {row[f'hr@{metric_k}']:7.4f} | "
            f"{row['mrr']:7.4f} | {row[f'precision@{metric_k}']:7.4f} | {row[f'ndcg@{metric_k}']:8.4f}"
        )


def _config_payload(config: RunConfig, embedding_config) -> dict[str, Any]:
    payload = {
        "dataset": config.dataset,
        "embedding": config.embedding,
        "embedding_provider": embedding_config.provider,
        "embedding_model": embedding_config.model,
        "strategies": [normalize_strategy_id(s) for s in config.strategies],
        "index_mode": config.index_mode,
        "params": config.params,
        "top_k": config.top_k,
        "metric_k": config.effective_metric_k,
    }
    if config.dynamic_overrides:
        payload["dynamic_overrides"] = dict(config.dynamic_overrides)
    return payload


def _save_result(result: dict[str, Any], config: RunConfig) -> Path:
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"benchmark_{artifacts.slugify(config.dataset)}_{config.index_mode}_{timestamp}.json"
    out_path = paths.RESULTS_DIR / name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["result_path"] = str(out_path)
    return out_path
