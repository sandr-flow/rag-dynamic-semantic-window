"""Per-domain hyperparameter tuning for dynamic_semantic (Optuna).

Core hypothesis of the project: the strategy's hyperparameters should be tuned
per domain. So tuning is keyed by (dataset, embedding) and its output is a
reusable ``tuned`` artifact the runner can apply.

The expensive part — sentence/question embeddings and similarity matrices — is
computed once into a cached corpus and reused across trials and re-tunes.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.config import DEFAULT_DYNAMIC_SEMANTIC_CONFIG
from src.corpus_data import (
    ArticleData,
    CorpusData,
    QuestionData,
    find_answer_sentence_idx,
    split_corpus_by_documents,
)
from src.utils import build_embedding_texts, split_into_sentences

from . import artifacts, paths
from .embeddings import prepare_embed_model, report_cache


def _neighbor_sims(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.array([])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    return np.sum(normalized[:-1] * normalized[1:], axis=1)


def _question_sims(q_embedding: np.ndarray, sentence_embeddings: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(q_embedding)
    if q_norm == 0:
        return np.zeros(len(sentence_embeddings))
    q = q_embedding / q_norm
    s_norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    s_norms = np.where(s_norms == 0, 1, s_norms)
    return (sentence_embeddings / s_norms) @ q


def _build_corpus(
    items: list[dict],
    embed_model,
    *,
    source: str,
    embedding_provider: str,
    embedding_model: str,
    phantom_window: int,
    top_k: int = 100,
    min_sentences: int = 10,
) -> CorpusData:
    """Precompute sentence/question embeddings + similarity matrices."""
    articles: list[ArticleData] = []
    total = len(items)
    log_stride = max(1, total // 10)
    for article_id, item in enumerate(items):
        sentences = split_into_sentences(item["text"])
        if len(sentences) < min_sentences:
            continue
        embedding_texts = build_embedding_texts(sentences, phantom_window)
        embeddings = np.array(
            embed_model.get_text_embedding_batch(embedding_texts), dtype=np.float32
        )
        articles.append(
            ArticleData(
                article_id=article_id,
                title=item.get("title", f"item_{article_id}"),
                sentences=sentences,
                embeddings=embeddings,
                neighbor_sims=_neighbor_sims(embeddings),
            )
        )
        if (article_id + 1) % log_stride == 0 or article_id + 1 == total:
            print(f"  [INFO] embedded {article_id + 1}/{total} articles")

    item_by_id = {i: item for i, item in enumerate(items)}

    # Collect all questions first, then embed them in one batched pass
    pending: list[tuple[ArticleData, dict]] = []
    for article in articles:
        for qa in item_by_id[article.article_id].get("qa_pairs", []):
            pending.append((article, qa))

    question_embeddings = (
        embed_model.get_text_embedding_batch([qa["question"] for _, qa in pending])
        if pending
        else []
    )

    questions: list[QuestionData] = []
    for qid, ((article, qa), raw_embedding) in enumerate(
        zip(pending, question_embeddings, strict=True)
    ):
        answer_sentence = qa.get("answer_sentence", qa.get("answer", ""))
        q_embedding = np.array(raw_embedding, dtype=np.float32)
        sentence_sims = _question_sims(q_embedding, article.embeddings)
        k = min(top_k, len(sentence_sims))
        top_k_indices = np.argsort(sentence_sims)[::-1][:k].astype(np.int32)
        questions.append(
            QuestionData(
                question_id=qid,
                article_id=article.article_id,
                question=qa["question"],
                answer_sentence=answer_sentence,
                answer_sentence_idx=find_answer_sentence_idx(article.sentences, answer_sentence),
                embedding=q_embedding,
                sentence_sims=sentence_sims,
                top_k_indices=top_k_indices,
            )
        )

    embed_dim = articles[0].embeddings.shape[1] if articles else 384
    return CorpusData(
        articles=articles,
        questions=questions,
        embed_dim=embed_dim,
        top_k=top_k,
        source=source,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        phantom_window=phantom_window,
        embedding_mode=_embedding_mode(phantom_window),
    )


def _embedding_mode(phantom_window: int) -> str:
    return f"phantom_w{phantom_window}" if phantom_window > 0 else "sentence"


def _corpus_cache_path(dataset: str, embedding: str, phantom_window: int) -> Path:
    key = f"{artifacts.tuned_key(dataset, embedding)}__{_embedding_mode(phantom_window)}"
    return paths.CORPUS_CACHE_DIR / f"{key}.pkl"


def _load_or_build_corpus(
    dataset: str,
    embedding: str,
    *,
    rebuild: bool,
    phantom_window: int,
) -> CorpusData:
    cache_path = _corpus_cache_path(dataset, embedding, phantom_window)
    if cache_path.exists() and not rebuild:
        print(f"[INFO] Reusing cached corpus: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    embed_model, embedding_config = prepare_embed_model(embedding)

    items = artifacts.load_dataset_items(dataset)
    print(
        f"[INFO] Building corpus cache for {dataset} + {embedding} "
        f"mode={_embedding_mode(phantom_window)} ({len(items)} docs)..."
    )
    corpus = _build_corpus(
        items,
        embed_model,
        source=dataset,
        embedding_provider=embedding_config.provider,
        embedding_model=embedding_config.model,
        phantom_window=phantom_window,
    )
    paths.CORPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(corpus, f)
    report_cache(embed_model)
    valid = sum(1 for q in corpus.questions if q.answer_sentence_idx >= 0)
    print(
        f"[OK] Corpus cached: {len(corpus.articles)} articles, "
        f"{len(corpus.questions)} questions ({valid} with answer found)"
    )
    return corpus


def tune(
    dataset: str,
    embedding: str,
    *,
    n_trials: int = 100,
    target_clusters: int = 5,
    soft_token_limit: int = 1200,
    rebuild_corpus: bool = False,
    phantom_window: int = DEFAULT_DYNAMIC_SEMANTIC_CONFIG.phantom_window,
    train_ratio: float = 0.70,
    split_seed: int = 42,
    hpo_config: str | None = None,
) -> dict:
    """Run per-domain Optuna HPO and save a tuned-params artifact."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ValueError("Optuna not installed. Run: pip install optuna") from exc

    from run_optuna import create_objective, evaluate_params  # reuse exact objective math
    from src.hpo_config import load_hpo_settings

    corpus = _load_or_build_corpus(
        dataset,
        embedding,
        rebuild=rebuild_corpus,
        phantom_window=phantom_window,
    )
    train_corpus, val_corpus = split_corpus_by_documents(
        corpus, train_ratio=train_ratio, seed=split_seed
    )
    valid_train_questions = sum(
        1 for q in train_corpus.questions if q.answer_sentence_idx >= 0
    )
    valid_val_questions = sum(
        1 for q in val_corpus.questions if q.answer_sentence_idx >= 0
    )
    if valid_train_questions == 0:
        raise ValueError(
            f"Corpus for '{dataset}' + '{embedding}' has no questions with a locatable "
            "answer sentence (documents may be too short for the min-sentence floor). "
            "Tune on a larger dataset (e.g. qasper/wikipedia)."
        )
    hpo_settings = load_hpo_settings(path=hpo_config, soft_token_limit=soft_token_limit)

    print(
        f"\n[INFO] Running {n_trials} Optuna trials for {dataset} + {embedding} "
        f"({len(train_corpus.articles)} train docs/{valid_train_questions} valid q, "
        f"{len(val_corpus.articles)} val docs/{valid_val_questions} valid q)..."
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    objective = create_objective(
        train_corpus,
        target_clusters=target_clusters,
        hpo_settings=hpo_settings,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    params = dict(best.params)
    params["phantom_window"] = phantom_window
    metrics_train = evaluate_params(
        train_corpus,
        best.params,
        target_clusters=target_clusters,
        hpo_settings=hpo_settings,
    )
    metrics_val = evaluate_params(
        val_corpus,
        best.params,
        target_clusters=target_clusters,
        hpo_settings=hpo_settings,
    )
    metrics = metrics_val if metrics_val["valid_questions"] else metrics_train
    target = artifacts.save_tuned(
        dataset,
        embedding,
        params,
        metrics,
        metrics_train=metrics_train,
        metrics_val=metrics_val,
        tuning={
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "target_clusters": target_clusters,
            "soft_token_limit": soft_token_limit,
            "embedding_mode": _embedding_mode(phantom_window),
            "phantom_window": phantom_window,
            "train_articles": len(train_corpus.articles),
            "val_articles": len(val_corpus.articles),
            # Scoring policy: hard token wall at soft_token_limit (an aligned
            # budget vs the fixed-window baseline), HR first, token savings a
            # strict sub-HR tie-breaker, MRR last. See ObjectivePolicy.
            "objective": asdict(hpo_settings.objective),
            "hpo_config": hpo_config,
        },
    )

    print("\n" + "=" * 64)
    print(f"[OK] Best trial #{best.number}: score={best.value:.4f} "
          f"train_HR={metrics_train['hit_rate']:.4f} "
          f"val_HR={metrics_val['hit_rate']:.4f} "
          f"val_MRR={metrics_val['mrr']:.4f} "
          f"val_tokens={metrics_val['avg_tokens']:.0f}")
    print("Best params:")
    for key, value in best.params.items():
        print(f"   {key}: {value}")
    print(f"\n[OK] Tuned artifact saved: {target}")
    print(f"     Use it with: --params tuned (dataset '{dataset}', embedding '{embedding}')")
    return {
        "params": params,
        "metrics": metrics,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
    }
