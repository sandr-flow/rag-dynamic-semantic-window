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

from src.embedding_cache import CachingEmbedding
from src.metrics import compute_all_metrics, compute_multi_answer_metrics
from src.significance import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    compare_to_baselines,
    format_comparisons,
)
from src.strategy_registry import (
    STRATEGY_OVERRIDE_KEYS,
    create_strategy,
    normalize_strategy_id,
)
from src.tokens import count_tokens

from . import artifacts, paths
from .corpus_filter import drop_unchunkable_items
from .embeddings import prepare_embed_model, report_cache
from .runconfig import RunConfig


def _resolve_strategy_overrides(
    config: RunConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Per-strategy param overrides and tuned manifests.

    ``params=tuned`` loads a tuned artifact for each requested strategy when
    one exists; missing artifacts keep library defaults. Manual
    ``dynamic_overrides`` still stack on top of dynamic_semantic.
    """
    overrides: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    strategy_ids = [normalize_strategy_id(s) for s in config.strategies]

    if config.params == "tuned":
        tuned_source = config.tuned_dataset or config.dataset
        for strategy_id in strategy_ids:
            tuned = artifacts.load_tuned(tuned_source, config.embedding, strategy=strategy_id)
            if tuned is None:
                continue
            manifests[strategy_id] = tuned
            overrides[strategy_id] = dict(tuned.get("params") or {})
        if "dynamic_semantic" in strategy_ids and "dynamic_semantic" not in manifests:
            raise ValueError(
                f"No tuned params for dataset '{tuned_source}' + embedding "
                f"'{config.embedding}'. Run: python -m stand tune ..."
            )

    if config.dynamic_overrides:
        allowed = set(STRATEGY_OVERRIDE_KEYS["dynamic_semantic"])
        unknown = sorted(set(config.dynamic_overrides) - allowed)
        if unknown:
            raise ValueError(
                "Unsupported dynamic_overrides keys: "
                f"{', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}"
            )
        overrides.setdefault("dynamic_semantic", {}).update(config.dynamic_overrides)

    return overrides, manifests


def _resolve_tuned(config: RunConfig) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Backward-compatible view: dynamic_semantic overrides + its manifest."""
    overrides, manifests = _resolve_strategy_overrides(config)
    return overrides.get("dynamic_semantic", {}), manifests.get("dynamic_semantic")


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


def _drop_questions_without_documents(
    eval_questions: list[dict[str, Any]],
    corpus_items: list[dict[str, Any]],
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Keep only questions whose every answer document is actually indexed.

    Unchunkable documents are dropped from the corpus, but their questions
    used to stay in the evaluation set. With no answer document in the index
    such a question scores zero for every strategy, so the whole comparison
    is shifted down by the share of dropped documents while looking like a
    genuine retrieval failure. Questions of unknown provenance (no
    ``source_docs``) are kept: silently dropping them would hide data, and
    the previous behaviour is the safer default there.
    """
    indexed = {str(item.get("id", idx)) for idx, item in enumerate(corpus_items)}
    kept = [
        qa
        for qa in eval_questions
        if all(str(doc) in indexed for doc in qa.get("source_docs", []))
    ]
    if verbose and len(kept) < len(eval_questions):
        dropped = len(eval_questions) - len(kept)
        print(
            f"  [INFO] dropped {dropped} question(s) whose answer document is "
            f"not indexed, {len(kept)} remain"
        )
    return kept


def _metrics_for_qa(
    texts: list[str], qa: dict[str, Any], metric_k: int
) -> dict[str, float]:
    if "answer_sentences" in qa:
        return compute_multi_answer_metrics(texts, qa["answer_sentences"], k=metric_k)
    answer = qa.get("answer_sentence", qa.get("answer", ""))
    return compute_all_metrics(texts, answer, k=metric_k)


EVAL_LOG_INTERVAL_S = 10.0


def _build_strategies(
    strategy_ids: list[str],
    documents: list[Document],
    top_k: int,
    strategy_overrides: dict[str, dict[str, Any]],
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
        started = time.time()
        if verbose:
            print(
                f"     {log_prefix}index {normalized} ({i}/{len(strategy_ids)})...",
                flush=True,
            )
        overrides = strategy_overrides.get(normalized, {})
        try:
            strategy = create_strategy(normalized, documents, top_k=top_k, overrides=overrides)
            strategies.append(strategy)
            if verbose:
                print(
                    f"     {log_prefix}index {normalized} done "
                    f"({time.time() - started:.1f}s)",
                    flush=True,
                )
        except Exception as exc:  # incompatible parser for this document type
            failed.add(normalized)
            print(f"  [skip] strategy '{normalized}' failed to build: "
                  f"{type(exc).__name__}: {exc}", flush=True)
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
    expected = total * len(strategies)
    started = time.time()
    last_log = 0.0
    for qi, qa in enumerate(qa_pairs, 1):
        now = time.time()
        # Time-based heartbeat: the first question and then at most one line
        # every EVAL_LOG_INTERVAL_S, so long evals never look hung.
        if verbose and (qi == 1 or qi == total or now - last_log >= EVAL_LOG_INTERVAL_S):
            last_log = now
            done = sum(len(rows) for rows in results.values())
            eta = ""
            if done:
                remaining_s = (expected - done) * (now - started) / done
                eta = f", ETA {remaining_s / 60:.1f} min"
            print(
                f"     {log_prefix}eval {qi}/{total} questions "
                f"({done}/{expected} strategy queries done{eta})",
                flush=True,
            )
        question = qa["question"]
        for strategy in strategies:
            nodes = strategy.retrieve(question)
            texts = [n.node.text for n in nodes]
            metrics = _metrics_for_qa(texts, qa, metric_k)
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
    strategy_overrides, tuned_manifests = _resolve_strategy_overrides(config)
    tuned_manifest = tuned_manifests.get("dynamic_semantic")

    embed_model, embedding_config = prepare_embed_model(config.embedding)

    if artifacts.is_extrahard(config.dataset) and config.index_mode != "shared":
        raise ValueError(
            f"Dataset '{config.dataset}' is extrahard (cross-document compound "
            "questions) and requires --index-mode shared"
        )

    corpus_items = artifacts.load_corpus_items(config.dataset, limit=config.limit)
    if not corpus_items:
        raise ValueError(f"Dataset '{config.dataset}' has no corpus items")

    n_before = len(corpus_items)
    corpus_items = drop_unchunkable_items(corpus_items, verbose=verbose)
    if verbose and len(corpus_items) < n_before:
        print(f"  [INFO] dropped {n_before - len(corpus_items)} unchunkable doc(s), {len(corpus_items)} remain")
    if not corpus_items:
        raise ValueError(f"Dataset '{config.dataset}' has no chunkable items left")

    eval_questions = artifacts.load_eval_questions(config.dataset, limit=config.limit)
    if not eval_questions:
        raise ValueError(f"Dataset '{config.dataset}' has no evaluation questions")

    eval_questions = _drop_questions_without_documents(
        eval_questions, corpus_items, verbose=verbose
    )
    if not eval_questions:
        raise ValueError(
            f"Dataset '{config.dataset}' has no questions whose answer document "
            "survived corpus filtering"
        )

    if verbose:
        print("=" * 64)
        print("Dynamic Semantic Window Benchmark")
        print("=" * 64)
        print(f"Dataset:    {config.dataset} ({len(corpus_items)} docs)")
        print(f"Embedding:  {embedding_config.provider}/{embedding_config.model}")
        print(f"Strategies: {', '.join(strategy_ids)}")
        print(f"Index mode: {config.index_mode}")
        print(f"Params:     {config.params}")
        print(f"Top-K: {config.top_k}, Metric-K: {metric_k}")
        if strategy_overrides:
            for sid, params in strategy_overrides.items():
                source = "tuned" if sid in tuned_manifests else "overrides"
                print(f"  {sid} ({source}): {params}")
        defaulted = [sid for sid in strategy_ids if sid not in strategy_overrides]
        if defaulted and config.params == "tuned":
            print(f"  library defaults (no tuned artifact): {', '.join(defaulted)}")
        if tuned_manifest and tuned_manifest.get("metrics_val"):
            val = tuned_manifest["metrics_val"]
            tuned_from = config.tuned_dataset or config.dataset
            print(
                f"Tuned params from '{tuned_from}' "
                f"(eval on '{config.dataset}'): "
                f"HR={val.get('hit_rate', 0):.4f}, "
                f"MRR={val.get('mrr', 0):.4f}, "
                f"tokens={val.get('avg_tokens', 0):.0f}"
            )
        print("-" * 64)

    start = time.time()
    aggregate: dict[str, list[dict]] = {}
    failed: set[str] = set()

    # Question embeddings are needed one-by-one during retrieval; batch the
    # cold ones up front so evaluation is not serialized on network calls.
    if isinstance(embed_model, CachingEmbedding):
        embed_model.prewarm_queries([qa["question"] for qa in eval_questions])

    multi_answer = any("answer_sentences" in qa for qa in eval_questions)

    if config.index_mode == "shared":
        documents = [_document_from_item(item, idx) for idx, item in enumerate(corpus_items)]
        shared_prefix = f"[shared {len(documents)} docs, {len(eval_questions)} q]"
        if verbose:
            print(f"  {shared_prefix} indexing...", flush=True)
        shared_start = time.time()
        strategies = _build_strategies(
            strategy_ids,
            documents,
            config.top_k,
            strategy_overrides,
            failed,
            verbose=verbose,
            log_prefix=f"{shared_prefix} ",
        )
        if verbose:
            print(f"  {shared_prefix} evaluating...", flush=True)
        _merge(
            aggregate,
            _evaluate(
                strategies,
                eval_questions,
                metric_k,
                verbose=verbose,
                log_prefix=f"{shared_prefix} ",
            ),
        )
        if verbose:
            print(f"  {shared_prefix} done ({time.time() - shared_start:.1f}s)", flush=True)
    else:  # per_document
        n_docs = len(corpus_items)
        for doc_idx, item in enumerate(corpus_items, 1):
            doc_qa = [qa for qa in eval_questions if str(item.get("id", doc_idx - 1)) in qa.get("source_docs", [])]
            if not doc_qa:
                doc_qa = item.get("qa_pairs", [])
            n_q = len(doc_qa)
            doc_prefix = f"[{doc_idx}/{n_docs}]"
            if verbose:
                print(f"  {doc_prefix} {item['title']} ({n_q} q) - indexing...", flush=True)
            doc_start = time.time()
            documents = [_document_from_item(item, doc_idx - 1)]
            strategies = _build_strategies(
                strategy_ids,
                documents,
                config.top_k,
                strategy_overrides,
                failed,
                verbose=verbose,
                log_prefix=f"{doc_prefix} ",
            )
            if verbose:
                print(f"     {doc_prefix} evaluating...", flush=True)
            _merge(
                aggregate,
                _evaluate(
                    strategies,
                    doc_qa,
                    metric_k,
                    verbose=verbose,
                    log_prefix=f"{doc_prefix} ",
                ),
            )
            if verbose:
                print(f"     {doc_prefix} done ({time.time() - doc_start:.1f}s)", flush=True)

    if failed and not aggregate:
        raise ValueError(
            "No strategy could be built for this dataset. Skipped: " + ", ".join(sorted(failed))
        )

    report_cache(embed_model, verbose=verbose)

    total_questions = len(eval_questions)
    summary = _summarize(aggregate, metric_k, multi_answer=multi_answer)
    metric_keys = [f"hr@{metric_k}", "mrr"]
    if multi_answer:
        metric_keys.append(f"partial_hr@{metric_k}")
    comparisons = compare_to_baselines(
        aggregate,
        target="Dynamic Semantic",
        metric_keys=metric_keys,
    )
    if verbose:
        _print_table(summary, metric_k, multi_answer=multi_answer)
        if comparisons:
            print()
            for line in format_comparisons(comparisons):
                print(line)
        print(f"\nTotal: {time.time() - start:.1f}s")

    result = {
        "config": _config_payload(config, embedding_config, strategy_overrides),
        "dataset": {"name": config.dataset, "docs": len(corpus_items), "questions": total_questions},
        "summary": summary,
        "aggregate": {name: [dict(m) for m in rows] for name, rows in aggregate.items()},
    }
    if comparisons:
        result["comparisons"] = {
            "target": "Dynamic Semantic",
            "n_resamples": DEFAULT_RESAMPLES,
            "confidence": DEFAULT_CONFIDENCE,
            "seed": DEFAULT_SEED,
            "rows": [comparison.to_dict() for comparison in comparisons],
        }
    if tuned_manifests:
        result["tuned_by_strategy"] = {
            sid: {
                "source_dataset": config.tuned_dataset or config.dataset,
                "created_at": manifest.get("created_at"),
                "params": manifest.get("params"),
                "metrics_train": manifest.get("metrics_train"),
                "metrics_val": manifest.get("metrics_val"),
                "tuning": manifest.get("tuning"),
            }
            for sid, manifest in tuned_manifests.items()
        }
    if tuned_manifest:
        result["tuned"] = {
            "source_dataset": config.tuned_dataset or config.dataset,
            "created_at": tuned_manifest.get("created_at"),
            "metrics_train": tuned_manifest.get("metrics_train"),
            "metrics_val": tuned_manifest.get("metrics_val"),
            "tuning": tuned_manifest.get("tuning"),
        }
    _save_result(result, config)
    return result


def _summarize(
    aggregate: dict[str, list[dict]], metric_k: int, *, multi_answer: bool = False
) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in aggregate.items():
        if not metrics:
            continue
        row = {
            "strategy": name,
            "tokens": float(np.mean([m["tokens"] for m in metrics])),
            f"hr@{metric_k}": float(np.mean([m[f"hr@{metric_k}"] for m in metrics])),
            "mrr": float(np.mean([m["mrr"] for m in metrics])),
            f"precision@{metric_k}": float(np.mean([m[f"precision@{metric_k}"] for m in metrics])),
            f"ndcg@{metric_k}": float(np.mean([m[f"ndcg@{metric_k}"] for m in metrics])),
        }
        if multi_answer:
            partial_key = f"partial_hr@{metric_k}"
            if partial_key in metrics[0]:
                row[partial_key] = float(np.mean([m[partial_key] for m in metrics]))
        rows.append(row)
    return rows


def _print_table(
    summary: list[dict[str, Any]], metric_k: int, *, multi_answer: bool = False
) -> None:
    if not summary:
        print("No results.")
        return
    suffix = f"@{metric_k}"
    partial_hdr = f" | {f'pHR{suffix}':>7}" if multi_answer else ""
    print("-" * (85 + len(partial_hdr)))
    print(
        f"{'Strategy':20} | {'Tokens':>7} | {f'HR{suffix}':>7} | "
        f"{'MRR':>7} | {f'P{suffix}':>7} | {f'NDCG{suffix}':>8}{partial_hdr}"
    )
    print("-" * (85 + len(partial_hdr)))
    for row in summary:
        partial_val = ""
        if multi_answer:
            partial_val = f" | {row.get(f'partial_hr@{metric_k}', 0.0):7.4f}"
        print(
            f"{row['strategy']:20} | {row['tokens']:7.1f} | {row[f'hr@{metric_k}']:7.4f} | "
            f"{row['mrr']:7.4f} | {row[f'precision@{metric_k}']:7.4f} | "
            f"{row[f'ndcg@{metric_k}']:8.4f}{partial_val}"
        )


def _config_payload(
    config: RunConfig,
    embedding_config,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "dataset": config.dataset,
        "embedding": config.embedding,
        "embedding_provider": embedding_config.provider,
        "embedding_model": embedding_config.model,
        "strategies": [normalize_strategy_id(s) for s in config.strategies],
        "index_mode": config.index_mode,
        "params": config.params,
        "strategy_overrides": {
            sid: dict(params) for sid, params in (strategy_overrides or {}).items()
        },
        "top_k": config.top_k,
        "metric_k": config.effective_metric_k,
    }
    if config.tuned_dataset:
        payload["tuned_dataset"] = config.tuned_dataset
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
