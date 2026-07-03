"""Per-domain hyperparameter tuning for dynamic_semantic (Optuna).

Core hypothesis of the project: the strategy's hyperparameters should be tuned
per domain. So tuning is keyed by (dataset, embedding) and its output is a
reusable ``tuned`` artifact the runner can apply.

The expensive part — sentence/question embeddings and similarity matrices — is
computed once into a cached corpus and reused across trials and re-tunes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from llama_index.core import Settings

from src.corpus_data import ArticleData, CorpusData, QuestionData, find_answer_sentence_idx
from src.providers import build_embedding_model, embedding_config_from_env
from src.utils import split_into_sentences

from . import artifacts, paths


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
    top_k: int = 100,
    min_sentences: int = 10,
) -> CorpusData:
    """Precompute sentence/question embeddings + similarity matrices."""
    articles: list[ArticleData] = []
    for article_id, item in enumerate(items):
        sentences = split_into_sentences(item["text"])
        if len(sentences) < min_sentences:
            continue
        embeddings = np.array(
            [embed_model.get_text_embedding(s) for s in sentences], dtype=np.float32
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

    item_by_id = {i: item for i, item in enumerate(items)}

    questions: list[QuestionData] = []
    qid = 0
    for article in articles:
        for qa in item_by_id[article.article_id].get("qa_pairs", []):
            answer_sentence = qa.get("answer_sentence", qa.get("answer", ""))
            q_embedding = np.array(
                embed_model.get_text_embedding(qa["question"]), dtype=np.float32
            )
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
            qid += 1

    embed_dim = articles[0].embeddings.shape[1] if articles else 384
    return CorpusData(
        articles=articles,
        questions=questions,
        embed_dim=embed_dim,
        top_k=top_k,
        source=source,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )


def _corpus_cache_path(dataset: str, embedding: str) -> Path:
    return paths.CORPUS_CACHE_DIR / f"{artifacts.tuned_key(dataset, embedding)}.pkl"


def _load_or_build_corpus(dataset: str, embedding: str, *, rebuild: bool) -> CorpusData:
    cache_path = _corpus_cache_path(dataset, embedding)
    if cache_path.exists() and not rebuild:
        print(f"[INFO] Reusing cached corpus: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    info = artifacts.get_embedding(embedding)
    if info is None:
        raise ValueError(f"Unknown embedding '{embedding}'. See: python -m stand list")
    embedding_config = embedding_config_from_env(
        provider=info.provider, model=info.model, api_key_env=info.api_key_env, base_url=info.base_url
    )
    embed_model = build_embedding_model(embedding_config)
    Settings.embed_model = embed_model

    items = artifacts.load_dataset_items(dataset)
    print(f"[INFO] Building corpus cache for {dataset} + {embedding} ({len(items)} docs)...")
    corpus = _build_corpus(
        items,
        embed_model,
        source=dataset,
        embedding_provider=embedding_config.provider,
        embedding_model=embedding_config.model,
    )
    paths.CORPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(corpus, f)
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
) -> dict:
    """Run per-domain Optuna HPO and save a tuned-params artifact."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ValueError("Optuna not installed. Run: pip install optuna") from exc

    from run_optuna import create_objective  # reuse the exact objective math
    from src.hpo_config import load_hpo_settings

    corpus = _load_or_build_corpus(dataset, embedding, rebuild=rebuild_corpus)
    valid_questions = sum(1 for q in corpus.questions if q.answer_sentence_idx >= 0)
    if valid_questions == 0:
        raise ValueError(
            f"Corpus for '{dataset}' + '{embedding}' has no questions with a locatable "
            "answer sentence (documents may be too short for the min-sentence floor). "
            "Tune on a larger dataset (e.g. qasper/wikipedia)."
        )
    hpo_settings = load_hpo_settings(soft_token_limit=soft_token_limit)

    print(f"\n[INFO] Running {n_trials} Optuna trials for {dataset} + {embedding}...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    objective = create_objective(corpus, target_clusters=target_clusters, hpo_settings=hpo_settings)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    metrics = {
        "score": best.value,
        "hit_rate": best.user_attrs.get("hit_rate", 0),
        "mrr": best.user_attrs.get("mrr", 0),
        "avg_tokens": best.user_attrs.get("avg_tokens", 0),
        "tokens_ok": best.user_attrs.get("tokens_ok", False),
    }
    target = artifacts.save_tuned(dataset, embedding, best.params, metrics)

    print("\n" + "=" * 64)
    print(f"[OK] Best trial #{best.number}: score={best.value:.4f} "
          f"HR={metrics['hit_rate']:.4f} MRR={metrics['mrr']:.4f} "
          f"tokens={metrics['avg_tokens']:.0f}")
    print("Best params:")
    for key, value in best.params.items():
        print(f"   {key}: {value}")
    print(f"\n[OK] Tuned artifact saved: {target}")
    print(f"     Use it with: --params tuned (dataset '{dataset}', embedding '{embedding}')")
    return {"params": best.params, "metrics": metrics}
