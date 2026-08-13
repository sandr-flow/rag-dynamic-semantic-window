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
    build_shared_global_arrays,
    find_answer_sentence_idx,
    question_is_valid,
    split_corpus_by_documents,
)
from src.seed_retrieval import dual_seed_indices
from src.utils import build_embedding_texts, split_into_sentences

from . import artifacts, paths
from .corpus_filter import drop_unchunkable_items
from .embeddings import prepare_embed_model, report_cache


def _neighbor_sims(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.array([])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    return np.sum(normalized[:-1] * normalized[1:], axis=1)


def _seed_indices(
    q_embedding: np.ndarray,
    phantom_embeddings: np.ndarray,
    clean_embeddings: np.ndarray | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Phantom query-sims (for expansion scores) plus dual-seed candidate order."""
    sentence_sims = _question_sims(q_embedding, phantom_embeddings)
    k = min(k, len(sentence_sims))
    clean_sims = None
    if clean_embeddings is not None:
        clean_sims = _question_sims(q_embedding, clean_embeddings)
    return sentence_sims, dual_seed_indices(sentence_sims, clean_sims, k)


def _question_sims(q_embedding: np.ndarray, sentence_embeddings: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(q_embedding)
    if q_norm == 0:
        return np.zeros(len(sentence_embeddings))
    q = q_embedding / q_norm
    s_norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    s_norms = np.where(s_norms == 0, 1, s_norms)
    return (sentence_embeddings / s_norms) @ q


def _embed_texts_batched(embed_model, texts: list[str], *, label: str) -> list:
    """Embed texts in chunks with periodic progress logs."""
    if not texts:
        return []
    batch_size = max(1, int(getattr(embed_model, "embed_batch_size", 64)))
    total = len(texts)
    log_stride = max(1, total // 10)
    print(f"  [INFO] embedding {total} {label}...", flush=True)
    embeddings: list = []
    for start in range(0, total, batch_size):
        chunk = texts[start : start + batch_size]
        embeddings.extend(embed_model.get_text_embedding_batch(chunk))
        done = min(start + len(chunk), total)
        if done % log_stride == 0 or done == total:
            print(f"  [INFO] embedded {done}/{total} {label}", flush=True)
    return embeddings


def _embed_articles(
    items: list[dict],
    embed_model,
    *,
    phantom_window: int,
    min_sentences: int = 10,
) -> list[ArticleData]:
    """Embed corpus documents into per-article arrays."""
    use_clean_adjacency = phantom_window > 0
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
        adjacency_embeddings = embeddings
        clean_embeddings = None
        if use_clean_adjacency:
            adjacency_embeddings = np.array(
                embed_model.get_text_embedding_batch(sentences), dtype=np.float32
            )
            clean_embeddings = adjacency_embeddings
        articles.append(
            ArticleData(
                article_id=article_id,
                title=item.get("title", f"item_{article_id}"),
                sentences=sentences,
                embeddings=embeddings,
                neighbor_sims=_neighbor_sims(adjacency_embeddings),
                clean_embeddings=clean_embeddings,
            )
        )
        if (article_id + 1) % log_stride == 0 or article_id + 1 == total:
            print(f"  [INFO] embedded {article_id + 1}/{total} articles")
    return articles


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
    """Precompute sentence/question embeddings + similarity matrices.

    Question sentence_sims are computed against the corpus embedding mode
    (phantom texts when phantom_window > 0), while per-article neighbor_sims
    always come from clean sentence embeddings — mirroring the live
    dual-space strategy so the HPO surrogate stays aligned. With
    phantom_window=0 the spaces coincide and no second batch is needed.
    """
    articles = _embed_articles(
        items, embed_model, phantom_window=phantom_window, min_sentences=min_sentences
    )
    item_by_id = {i: item for i, item in enumerate(items)}

    # Collect all questions first, then embed them in one batched pass
    pending: list[tuple[ArticleData, dict]] = []
    for article in articles:
        for qa in item_by_id[article.article_id].get("qa_pairs", []):
            pending.append((article, qa))

    question_embeddings = _embed_texts_batched(
        embed_model, [qa["question"] for _, qa in pending], label="questions"
    )

    questions: list[QuestionData] = []
    for qid, ((article, qa), raw_embedding) in enumerate(
        zip(pending, question_embeddings, strict=True)
    ):
        answer_sentence = qa.get("answer_sentence", qa.get("answer", ""))
        q_embedding = np.array(raw_embedding, dtype=np.float32)
        sentence_sims, top_k_indices = _seed_indices(
            q_embedding, article.embeddings, article.clean_embeddings, top_k
        )
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
        adjacency_space="clean",
    )


def _build_shared_corpus(
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
    """Precompute a shared-index corpus for ordinary single-answer datasets.

    Same documents and questions as :func:`_build_corpus`, but query
    similarities are computed against the whole corpus rather than the source
    article, so HPO optimizes under the cross-document competition the shared
    benchmark actually measures. Questions keep ``source_article_ids`` empty;
    ``global_answer_indices`` then resolves their single answer through
    ``article_offsets``.
    """
    from src.config import DEFAULT_EXPANSION_CONFIG

    articles = _embed_articles(
        items, embed_model, phantom_window=phantom_window, min_sentences=min_sentences
    )
    if not articles:
        raise ValueError("Shared tuning needs at least one chunkable corpus document")

    (
        global_sentences,
        global_neighbor_sims,
        global_garbage_mask,
        global_segment_ids,
        article_offsets,
    ) = build_shared_global_arrays(
        articles,
        [items[article.article_id] for article in articles],
        min_chunk_length=DEFAULT_EXPANSION_CONFIG.min_chunk_length,
    )
    global_embeddings = np.concatenate([article.embeddings for article in articles], axis=0)
    global_clean_embeddings = np.concatenate(
        [
            article.clean_embeddings if article.clean_embeddings is not None else article.embeddings
            for article in articles
        ],
        axis=0,
    )
    print(
        f"  [INFO] shared index: {len(global_sentences)} sentences "
        f"across {len(articles)} docs",
        flush=True,
    )

    pending: list[tuple[ArticleData, dict]] = []
    for article in articles:
        for qa in items[article.article_id].get("qa_pairs", []):
            pending.append((article, qa))

    question_embeddings = _embed_texts_batched(
        embed_model, [qa["question"] for _, qa in pending], label="questions"
    )

    questions: list[QuestionData] = []
    total_pending = len(pending)
    sim_log_stride = max(1, total_pending // 10)
    if total_pending:
        print(
            f"  [INFO] computing global similarities for {total_pending} questions...",
            flush=True,
        )
    for qid, ((article, qa), raw_embedding) in enumerate(
        zip(pending, question_embeddings, strict=True)
    ):
        answer_sentence = qa.get("answer_sentence", qa.get("answer", ""))
        q_embedding = np.array(raw_embedding, dtype=np.float32)
        sentence_sims, top_k_indices = _seed_indices(
            q_embedding, global_embeddings, global_clean_embeddings, top_k
        )
        questions.append(
            QuestionData(
                question_id=qid,
                article_id=article.article_id,
                question=qa["question"],
                answer_sentence=answer_sentence,
                # Local index; global_answer_indices adds the article offset.
                answer_sentence_idx=find_answer_sentence_idx(
                    article.sentences, answer_sentence
                ),
                embedding=q_embedding,
                sentence_sims=sentence_sims,
                top_k_indices=top_k_indices,
            )
        )
        if (qid + 1) % sim_log_stride == 0 or qid + 1 == total_pending:
            print(f"  [INFO] similarities {qid + 1}/{total_pending} questions", flush=True)

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
        adjacency_space="clean",
        kind="shared",
        global_sentences=global_sentences,
        global_neighbor_sims=global_neighbor_sims,
        global_garbage_mask=global_garbage_mask,
        global_segment_ids=global_segment_ids,
        article_offsets=article_offsets,
    )


def _build_extrahard_corpus(
    items: list[dict],
    pairs: list[dict],
    embed_model,
    *,
    source: str,
    embedding_provider: str,
    embedding_model: str,
    phantom_window: int,
    top_k: int = 100,
    min_sentences: int = 10,
) -> CorpusData:
    """Precompute a shared-index corpus for cross-document compound questions."""
    from src.config import DEFAULT_EXPANSION_CONFIG

    articles = _embed_articles(
        items, embed_model, phantom_window=phantom_window, min_sentences=min_sentences
    )
    if len(articles) < 2:
        raise ValueError("Extrahard tuning needs at least two chunkable corpus documents")

    (
        global_sentences,
        global_neighbor_sims,
        global_garbage_mask,
        global_segment_ids,
        article_offsets,
    ) = build_shared_global_arrays(
        articles,
        [items[article.article_id] for article in articles],
        min_chunk_length=DEFAULT_EXPANSION_CONFIG.min_chunk_length,
    )
    global_embeddings = np.concatenate([article.embeddings for article in articles], axis=0)
    global_clean_embeddings = np.concatenate(
        [
            article.clean_embeddings if article.clean_embeddings is not None else article.embeddings
            for article in articles
        ],
        axis=0,
    )
    print(
        f"  [INFO] shared index: {len(global_sentences)} sentences "
        f"across {len(articles)} docs",
        flush=True,
    )
    article_by_id = {article.article_id: article for article in articles}
    item_id_to_article = {
        str(item.get("id", idx)): idx for idx, item in enumerate(items)
    }

    pending: list[tuple[dict, list[int], list[int]]] = []
    for pair in pairs:
        source_article_ids: list[int] = []
        answer_local_indices: list[int] = []
        valid_pair = True
        for doc_id, answer in zip(
            pair["source_docs"], pair["answer_sentences"], strict=True
        ):
            article_id = item_id_to_article.get(str(doc_id))
            if article_id is None or article_id not in article_by_id:
                valid_pair = False
                break
            local_idx = find_answer_sentence_idx(
                article_by_id[article_id].sentences, answer
            )
            source_article_ids.append(article_id)
            answer_local_indices.append(local_idx)
        if valid_pair:
            pending.append((pair, source_article_ids, answer_local_indices))

    question_embeddings = _embed_texts_batched(
        embed_model,
        [pair["question"] for pair, _, _ in pending],
        label="compound questions",
    )

    questions: list[QuestionData] = []
    total_pending = len(pending)
    sim_log_stride = max(1, total_pending // 10)
    if total_pending:
        print(
            f"  [INFO] computing global similarities for {total_pending} compound questions...",
            flush=True,
        )
    for qid, ((pair, source_article_ids, answer_local_indices), raw_embedding) in enumerate(
        zip(pending, question_embeddings, strict=True)
    ):
        q_embedding = np.array(raw_embedding, dtype=np.float32)
        sentence_sims, top_k_indices = _seed_indices(
            q_embedding, global_embeddings, global_clean_embeddings, top_k
        )
        questions.append(
            QuestionData(
                question_id=qid,
                article_id=source_article_ids[0],
                question=pair["question"],
                answer_sentence=pair["answer_sentences"][0],
                answer_sentence_idx=answer_local_indices[0],
                embedding=q_embedding,
                sentence_sims=sentence_sims,
                top_k_indices=top_k_indices,
                source_article_ids=source_article_ids,
                answer_local_indices=answer_local_indices,
            )
        )
        if (qid + 1) % sim_log_stride == 0 or qid + 1 == total_pending:
            print(
                f"  [INFO] similarities {qid + 1}/{total_pending} compound questions",
                flush=True,
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
        adjacency_space="clean",
        kind="shared",
        global_sentences=global_sentences,
        global_neighbor_sims=global_neighbor_sims,
        global_garbage_mask=global_garbage_mask,
        global_segment_ids=global_segment_ids,
        article_offsets=article_offsets,
    )


def _embedding_mode(phantom_window: int) -> str:
    """Corpus embedding mode, also the corpus-cache key component.

    The ``__adj_clean`` suffix is kept so corpora built before clean-only
    adjacency (phantom-adjacency neighbor_sims under the plain
    ``phantom_wN`` key) are never silently reused. ``__dual_seed`` invalidates
    caches whose ``top_k_indices`` were phantom-only. With phantom_window=0
    the spaces coincide and both suffixes are meaningless.
    """
    if phantom_window <= 0:
        return "sentence"
    return f"phantom_w{phantom_window}__adj_clean__dual_seed"


def _corpus_cache_path(
    dataset: str, embedding: str, phantom_window: int, index_mode: str = "per_document"
) -> Path:
    key = f"{artifacts.tuned_key(dataset, embedding)}__{_embedding_mode(phantom_window)}"
    # Shared corpora carry different question sims; keep them in their own file
    # so a per-document cache is never silently reused (and vice versa).
    if index_mode == "shared":
        key = f"{key}__shared"
    return paths.CORPUS_CACHE_DIR / f"{key}.pkl"


def _load_or_build_corpus(
    dataset: str,
    embedding: str,
    *,
    rebuild: bool,
    phantom_window: int,
    index_mode: str = "per_document",
) -> CorpusData:
    cache_path = _corpus_cache_path(dataset, embedding, phantom_window, index_mode)
    if cache_path.exists() and not rebuild:
        print(f"[INFO] Reusing cached corpus: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    embed_model, embedding_config = prepare_embed_model(embedding)

    extrahard = artifacts.is_extrahard(dataset)
    if not extrahard and index_mode == "shared":
        items = drop_unchunkable_items(artifacts.load_dataset_items(dataset))
        print(
            f"[INFO] Building shared corpus for {dataset} + {embedding} "
            f"mode={_embedding_mode(phantom_window)} ({len(items)} docs)..."
        )
        corpus = _build_shared_corpus(
            items,
            embed_model,
            source=dataset,
            embedding_provider=embedding_config.provider,
            embedding_model=embedding_config.model,
            phantom_window=phantom_window,
        )
    elif extrahard:
        items = drop_unchunkable_items(artifacts.load_corpus_items(dataset))
        pairs = artifacts.load_eval_questions(dataset)
        print(
            f"[INFO] Building extrahard shared corpus for {dataset} + {embedding} "
            f"mode={_embedding_mode(phantom_window)} "
            f"({len(items)} docs, {len(pairs)} compound q)..."
        )
        corpus = _build_extrahard_corpus(
            items,
            pairs,
            embed_model,
            source=dataset,
            embedding_provider=embedding_config.provider,
            embedding_model=embedding_config.model,
            phantom_window=phantom_window,
        )
    else:
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
    valid = sum(1 for q in corpus.questions if question_is_valid(q, corpus))
    print(
        f"[OK] Corpus cached: {len(corpus.articles)} articles, "
        f"{len(corpus.questions)} questions ({valid} with answer found)"
    )
    return corpus


def _report_adjacency_distribution(corpus: CorpusData) -> None:
    """Print adjacency-cosine percentiles so threshold search bounds can be sanity-checked.

    Clean-space adjacency cosines sit lower and spread wider than phantom-space
    ones; if the mass falls below the configured threshold search range, the
    range must be widened via --hpo-config before burning trials.
    """
    sims = [article.neighbor_sims for article in corpus.articles if len(article.neighbor_sims)]
    if not sims:
        return
    values = np.concatenate(sims)
    p = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    space = getattr(corpus, "adjacency_space", "phantom")
    print(
        f"[INFO] adjacency sims ({space}, n={len(values)}): "
        f"p1={p[0]:.3f} p5={p[1]:.3f} p25={p[2]:.3f} p50={p[3]:.3f} "
        f"p75={p[4]:.3f} p95={p[5]:.3f} p99={p[6]:.3f}"
    )


def tune(
    dataset: str,
    embedding: str,
    *,
    strategy: str = "dynamic_semantic",
    n_trials: int = 100,
    target_clusters: int = 5,
    soft_token_limit: int = 1200,
    rebuild_corpus: bool = False,
    phantom_window: int = DEFAULT_DYNAMIC_SEMANTIC_CONFIG.phantom_window,
    train_ratio: float = 0.70,
    split_seed: int = 42,
    hpo_config: str | None = None,
    index_mode: str = "per_document",
    top_k: int = 5,
) -> dict:
    """Run per-domain Optuna HPO and save a tuned-params artifact."""
    from src.strategy_registry import normalize_strategy_id

    strategy_id = normalize_strategy_id(strategy)
    if strategy_id != "dynamic_semantic":
        from .tune_live import tune_live

        return tune_live(
            dataset,
            embedding,
            strategy_id,
            n_trials=n_trials,
            top_k=top_k,
            soft_token_limit=soft_token_limit,
            train_ratio=train_ratio,
            split_seed=split_seed,
            hpo_config=hpo_config,
            index_mode=index_mode,
        )

    return _tune_dynamic(
        dataset,
        embedding,
        n_trials=n_trials,
        target_clusters=target_clusters,
        soft_token_limit=soft_token_limit,
        rebuild_corpus=rebuild_corpus,
        phantom_window=phantom_window,
        train_ratio=train_ratio,
        split_seed=split_seed,
        hpo_config=hpo_config,
        index_mode=index_mode,
    )


def _tune_dynamic(
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
    index_mode: str = "per_document",
) -> dict:
    """Run per-domain Optuna HPO and save a tuned-params artifact."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ValueError("Optuna not installed. Run: pip install optuna") from exc

    from run_optuna import create_objective, evaluate_params  # reuse exact objective math
    from src.hpo_config import load_hpo_settings, quantize_params

    corpus = _load_or_build_corpus(
        dataset,
        embedding,
        rebuild=rebuild_corpus,
        phantom_window=phantom_window,
        index_mode=index_mode,
    )
    _report_adjacency_distribution(corpus)
    train_corpus, val_corpus = split_corpus_by_documents(
        corpus, train_ratio=train_ratio, seed=split_seed
    )
    valid_train_questions = sum(
        1 for q in train_corpus.questions if question_is_valid(q, train_corpus)
    )
    valid_val_questions = sum(
        1 for q in val_corpus.questions if question_is_valid(q, val_corpus)
    )
    if valid_train_questions == 0:
        raise ValueError(
            f"Corpus for '{dataset}' + '{embedding}' has no questions with a locatable "
            "answer sentence (documents may be too short for the min-sentence floor). "
            "Tune on a larger dataset (e.g. qasper/wikipedia)."
        )
    hpo_settings = load_hpo_settings(
        path=hpo_config, soft_token_limit=soft_token_limit, strategy="dynamic_semantic"
    )

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
    params = quantize_params(dict(best.params), hpo_settings.search_space)
    params["phantom_window"] = phantom_window
    params["dual_seed"] = True
    metrics_train = evaluate_params(
        train_corpus,
        params,
        target_clusters=target_clusters,
        hpo_settings=hpo_settings,
    )
    metrics_val = evaluate_params(
        val_corpus,
        params,
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
            # Scoring policy: soft token term around soft_token_limit (a
            # reference vs the fixed-window baseline, not a veto). Under it,
            # token savings are a sub-HR tie-breaker; above it the penalty
            # accelerates with excess. See ObjectivePolicy.
            "objective": asdict(hpo_settings.objective),
            "hpo_config": hpo_config,
            "corpus_kind": corpus.kind,
            "index_mode": "shared" if corpus.kind == "shared" else "per_document",
            "strategy": "dynamic_semantic",
            "path": "cached",
        },
    )

    print("\n" + "=" * 64)
    print(f"[OK] Best trial #{best.number}: score={best.value:.4f} "
          f"train_HR={metrics_train['hit_rate']:.4f} "
          f"val_HR={metrics_val['hit_rate']:.4f} "
          f"val_MRR={metrics_val['mrr']:.4f} "
          f"val_tokens={metrics_val['avg_tokens']:.0f}")
    print("Best params:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    print(f"\n[OK] Tuned artifact saved: {target}")
    print(f"     Use it with: --params tuned (dataset '{dataset}', embedding '{embedding}')")
    return {
        "params": params,
        "metrics": metrics,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
    }
