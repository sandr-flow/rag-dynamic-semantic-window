"""Live-path Optuna HPO for non-dynamic retrieval strategies.

Unlike the cached dynamic_semantic surrogate, baseline splitters change the
index itself (chunk size, window, semantic breakpoints). Each trial rebuilds
the strategy over the train documents and runs the same retrieve + metrics
path as ``stand run``. Sentence/chunk embeddings are reused through the
shared disk cache.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from src.hpo_config import (
    HPOSettings,
    compute_objective_score,
    load_hpo_settings,
    quantize_params,
    suggest_params,
)
from src.strategy_registry import normalize_strategy_id

from . import artifacts
from .corpus_filter import drop_unchunkable_items
from .embeddings import prepare_embed_model, report_cache
from .runner import (
    _build_strategies,
    _document_from_item,
    _drop_questions_without_documents,
    _evaluate,
)


def split_items_by_documents(
    items: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Document-level 70/30 split matching the dynamic HPO protocol.

    A question stays in a split only when every ``source_docs`` id belongs to
    that split (compound extrahard pairs cannot leak a held-out article).
    """
    if len(items) < 2:
        return items, questions, items, questions

    rng = np.random.default_rng(seed)
    order = np.arange(len(items))
    rng.shuffle(order)
    train_size = round(len(items) * train_ratio)
    train_size = max(1, min(len(items) - 1, train_size))
    train_idx = set(int(i) for i in order[:train_size])

    train_items = [items[i] for i in range(len(items)) if i in train_idx]
    val_items = [items[i] for i in range(len(items)) if i not in train_idx]
    id_by_pos = {i: str(item.get("id", i)) for i, item in enumerate(items)}
    train_ids = {id_by_pos[i] for i in train_idx}
    val_ids = {id_by_pos[i] for i in range(len(items)) if i not in train_idx}

    train_questions = [q for q in questions if _question_in_split(q, train_ids)]
    val_questions = [q for q in questions if _question_in_split(q, val_ids)]
    return train_items, train_questions, val_items, val_questions


def _question_in_split(question: dict[str, Any], doc_ids: set[str]) -> bool:
    source_docs = question.get("source_docs") or []
    if not source_docs:
        return False
    return all(str(doc_id) in doc_ids for doc_id in source_docs)


def evaluate_live_params(
    *,
    strategy_id: str,
    items: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    params: dict[str, Any],
    top_k: int,
    metric_k: int,
    index_mode: str,
    hpo_settings: HPOSettings,
) -> dict[str, Any]:
    """Build one strategy and score it on ``questions`` with the HPO objective."""
    strategy_id = normalize_strategy_id(strategy_id)
    questions = _drop_questions_without_documents(questions, items, verbose=False)
    if not questions:
        return {
            "score": hpo_settings.objective.invalid_score,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "avg_tokens": 0.0,
            "tokens_ok": False,
            "valid_questions": 0,
            "total_questions": 0,
        }

    overrides = {strategy_id: dict(params)}
    failed: set[str] = set()
    rows: list[dict[str, Any]] = []

    if index_mode == "shared":
        documents = [_document_from_item(item, idx) for idx, item in enumerate(items)]
        strategies = _build_strategies(
            [strategy_id],
            documents,
            top_k,
            overrides,
            failed,
            verbose=False,
        )
        if not strategies:
            raise ValueError(
                f"Strategy '{strategy_id}' failed to build: {', '.join(sorted(failed)) or 'unknown'}"
            )
        by_name = _evaluate(strategies, questions, metric_k, verbose=False)
        rows = next(iter(by_name.values()), [])
    else:
        for doc_idx, item in enumerate(items):
            doc_id = str(item.get("id", doc_idx))
            doc_qa = [qa for qa in questions if doc_id in [str(d) for d in qa.get("source_docs", [])]]
            if not doc_qa:
                continue
            documents = [_document_from_item(item, doc_idx)]
            strategies = _build_strategies(
                [strategy_id],
                documents,
                top_k,
                overrides,
                failed,
                verbose=False,
            )
            if not strategies:
                continue
            by_name = _evaluate(strategies, doc_qa, metric_k, verbose=False)
            rows.extend(next(iter(by_name.values()), []))

    if not rows:
        return {
            "score": hpo_settings.objective.invalid_score,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "avg_tokens": 0.0,
            "tokens_ok": False,
            "valid_questions": 0,
            "total_questions": len(questions),
        }

    hr_key = f"hr@{metric_k}"
    avg_hr = float(np.mean([row[hr_key] for row in rows]))
    avg_mrr = float(np.mean([row["mrr"] for row in rows]))
    avg_tokens = float(np.mean([row["tokens"] for row in rows]))
    score, tokens_ok = compute_objective_score(
        avg_hr=avg_hr,
        avg_mrr=avg_mrr,
        avg_tokens=avg_tokens,
        num_valid_questions=len(rows),
        policy=hpo_settings.objective,
    )
    return {
        "score": score,
        "hit_rate": avg_hr,
        "mrr": avg_mrr,
        "avg_tokens": avg_tokens,
        "tokens_ok": tokens_ok,
        "valid_questions": len(rows),
        "total_questions": len(questions),
    }


def create_live_objective(
    *,
    strategy_id: str,
    items: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    top_k: int,
    metric_k: int,
    index_mode: str,
    hpo_settings: HPOSettings,
):
    """Optuna objective that rebuilds the strategy each trial."""

    def objective(trial) -> float:
        params = suggest_params(trial, hpo_settings.search_space)
        metrics = evaluate_live_params(
            strategy_id=strategy_id,
            items=items,
            questions=questions,
            params=params,
            top_k=top_k,
            metric_k=metric_k,
            index_mode=index_mode,
            hpo_settings=hpo_settings,
        )
        trial.set_user_attr("hit_rate", metrics["hit_rate"])
        trial.set_user_attr("mrr", metrics["mrr"])
        trial.set_user_attr("avg_tokens", metrics["avg_tokens"])
        trial.set_user_attr("score", metrics["score"])
        trial.set_user_attr("tokens_ok", metrics["tokens_ok"])
        return metrics["score"]

    return objective


def tune_live(
    dataset: str,
    embedding: str,
    strategy: str,
    *,
    n_trials: int = 50,
    top_k: int = 5,
    soft_token_limit: int = 1200,
    train_ratio: float = 0.70,
    split_seed: int = 42,
    hpo_config: str | None = None,
    index_mode: str = "per_document",
) -> dict[str, Any]:
    """Tune one baseline strategy on the live retrieve path and save the artifact."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ValueError("Optuna not installed. Run: pip install optuna") from exc

    strategy_id = normalize_strategy_id(strategy)
    if strategy_id == "dynamic_semantic":
        raise ValueError("dynamic_semantic uses the cached corpus path in stand.tune")

    if artifacts.is_extrahard(dataset) and index_mode != "shared":
        raise ValueError(
            f"Dataset '{dataset}' is extrahard and requires --index-mode shared"
        )

    corpus_items = drop_unchunkable_items(artifacts.load_corpus_items(dataset))
    if not corpus_items:
        raise ValueError(f"Dataset '{dataset}' has no chunkable items")
    questions = artifacts.load_eval_questions(dataset)
    questions = _drop_questions_without_documents(questions, corpus_items, verbose=False)
    if not questions:
        raise ValueError(f"Dataset '{dataset}' has no evaluation questions")

    train_items, train_questions, val_items, val_questions = split_items_by_documents(
        corpus_items,
        questions,
        train_ratio=train_ratio,
        seed=split_seed,
    )
    if not train_questions:
        raise ValueError(
            f"Train split for '{dataset}' has no questions; check source_docs tagging"
        )

    prepare_embed_model(embedding)
    hpo_settings = load_hpo_settings(
        path=hpo_config,
        soft_token_limit=soft_token_limit,
        strategy=strategy_id,
    )
    metric_k = top_k

    print(
        f"\n[INFO] Running {n_trials} live Optuna trials for {strategy_id} "
        f"on {dataset} + {embedding} ({index_mode}; "
        f"{len(train_items)} train docs/{len(train_questions)} q, "
        f"{len(val_items)} val docs/{len(val_questions)} q)..."
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    objective = create_live_objective(
        strategy_id=strategy_id,
        items=train_items,
        questions=train_questions,
        top_k=top_k,
        metric_k=metric_k,
        index_mode=index_mode,
        hpo_settings=hpo_settings,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    params = quantize_params(dict(best.params), hpo_settings.search_space)
    metrics_train = evaluate_live_params(
        strategy_id=strategy_id,
        items=train_items,
        questions=train_questions,
        params=params,
        top_k=top_k,
        metric_k=metric_k,
        index_mode=index_mode,
        hpo_settings=hpo_settings,
    )
    metrics_val = (
        evaluate_live_params(
            strategy_id=strategy_id,
            items=val_items,
            questions=val_questions,
            params=params,
            top_k=top_k,
            metric_k=metric_k,
            index_mode=index_mode,
            hpo_settings=hpo_settings,
        )
        if val_questions
        else dict(metrics_train)
    )
    metrics = metrics_val if metrics_val["valid_questions"] else metrics_train
    target = artifacts.save_tuned(
        dataset,
        embedding,
        params,
        metrics,
        strategy=strategy_id,
        metrics_train=metrics_train,
        metrics_val=metrics_val,
        tuning={
            "strategy": strategy_id,
            "path": "live",
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "top_k": top_k,
            "soft_token_limit": soft_token_limit,
            "train_articles": len(train_items),
            "val_articles": len(val_items),
            "objective": asdict(hpo_settings.objective),
            "hpo_config": hpo_config,
            "index_mode": index_mode,
            "search_space": {key: asdict(spec) for key, spec in hpo_settings.search_space.items()},
        },
    )
    from llama_index.core import Settings

    report_cache(Settings.embed_model)

    print("\n" + "=" * 64)
    print(
        f"[OK] {strategy_id} best trial #{best.number}: score={best.value:.4f} "
        f"train_HR={metrics_train['hit_rate']:.4f} "
        f"val_HR={metrics_val['hit_rate']:.4f} "
        f"val_MRR={metrics_val['mrr']:.4f} "
        f"val_tokens={metrics_val['avg_tokens']:.0f}"
    )
    print("Best params:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    print(f"\n[OK] Tuned artifact saved: {target}")
    return {
        "strategy": strategy_id,
        "params": params,
        "metrics": metrics,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
    }


__all__ = [
    "create_live_objective",
    "evaluate_live_params",
    "split_items_by_documents",
    "tune_live",
]
