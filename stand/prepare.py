"""Prep layer: build reusable dataset and embedding artifacts (heavy, run once).

These are the universal loaders. They hit the network / LLM / model weights
once and write artifacts that the runner and tuner reuse offline.
"""

from __future__ import annotations

import asyncio
import os

from src.benchmark_datasets import load_benchmark_dataset
from src.providers import DEFAULT_API_KEY_ENVS, build_embedding_model, embedding_config_from_env

from . import artifacts


def _apply_qa_model_env(provider: str | None, model: str | None) -> str:
    """Point question_generator at the chosen QA model and return its label."""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL"] = model
    from src.providers import llm_config_from_env

    config = llm_config_from_env(provider=provider, model=model)
    return f"{config.provider}/{config.model}"


async def _generate_qa(items: list[dict], questions_per_item: int, qa_delay: float) -> list[dict]:
    """Generate QA pairs for items lacking them (rate-limited)."""
    from src.question_generator import generate_qa_pairs_async

    prepared = []
    for idx, item in enumerate(items):
        title = item.get("title", item.get("id", idx))
        qa_pairs = await generate_qa_pairs_async(item["text"], num_questions=questions_per_item)
        if qa_pairs:
            prepared.append({**item, "qa_pairs": qa_pairs})
            print(f"  [OK] [{len(prepared)}/{len(items)}] QA for: {title}")
        else:
            print(f"  [WARN] [{idx + 1}/{len(items)}] No valid QA for: {title}")
        if idx < len(items) - 1:
            await asyncio.sleep(max(0.0, qa_delay))
    return prepared


async def _prepare_dataset_async(
    *,
    source: str,
    name: str,
    num_articles: int | None,
    min_length: int,
    questions_per_article: int,
    split: str,
    skip: int,
    qa_provider: str | None,
    qa_model: str | None,
    qa_delay: float,
    dataset_path: str | None,
) -> artifacts.DatasetInfo:
    if source == "custom":
        if not dataset_path:
            raise ValueError("custom source requires --dataset-path")
        items = load_benchmark_dataset(dataset_path=dataset_path)
        qa_label = "custom"
        print(f"[OK] Loaded {len(items)} custom items (QA already present)")
    elif source == "financebench":
        from src.financebench_loader import load_financebench_items

        # num_articles keeps its CLI default of 5 for the generation sources;
        # for FinanceBench "all filings" is the sensible default, so only an
        # explicit positive --num-articles limits the document count.
        max_docs = num_articles if num_articles and num_articles > 0 else None
        scope = f"first {max_docs} filings" if max_docs else "all filings"
        print(f"[INFO] Loading FinanceBench open subset ({scope}, questions + PDFs)...")
        items = load_financebench_items(max_docs=max_docs)
        qa_label = "financebench:human"
        print(f"[OK] Loaded {len(items)} filings with human-annotated QA")
    else:
        qa_label = _apply_qa_model_env(qa_provider, qa_model)
        num_articles = num_articles or 5
        if source == "wikipedia":
            from src.wikipedia_loader import fetch_random_articles_batch

            print(f"[INFO] Fetching {num_articles} Wikipedia articles (min {min_length} chars)...")
            raw = await fetch_random_articles_batch(count=num_articles, min_length=min_length)
        elif source == "qasper":
            from src.qasper_loader import fetch_qasper_articles

            print(f"[INFO] Loading {num_articles} QASPER articles (split={split}, skip={skip})...")
            raw = fetch_qasper_articles(num_articles, min_length, split=split, skip=skip)
        else:
            raise ValueError(f"Unknown source: {source}")

        print(f"[OK] Got {len(raw)} articles")
        base_items = [{"id": i, "title": title, "text": text} for i, (title, text) in enumerate(raw)]
        print(f"[INFO] Generating QA with {qa_label} ({questions_per_article}/article)...")
        items = await _generate_qa(base_items, questions_per_article, qa_delay)
        print(f"[OK] Generated QA for {len(items)} articles")

    if not items:
        raise ValueError("No items with QA pairs were produced")

    info = artifacts.save_dataset(name, items, source=source, qa_model=qa_label)
    print(f"\n[OK] Dataset artifact saved: {info.label()}")
    print(f"     {artifacts.dataset_dir(name)}")
    return info


def prepare_dataset(
    *,
    source: str,
    name: str,
    num_articles: int | None = None,
    min_length: int = 2000,
    questions_per_article: int = 3,
    split: str = "validation",
    skip: int = 0,
    qa_provider: str | None = None,
    qa_model: str | None = None,
    qa_delay: float = 1.1,
    dataset_path: str | None = None,
) -> artifacts.DatasetInfo:
    """Build and persist a reusable dataset artifact."""
    return asyncio.run(
        _prepare_dataset_async(
            source=source,
            name=name,
            num_articles=num_articles,
            min_length=min_length,
            questions_per_article=questions_per_article,
            split=split,
            skip=skip,
            qa_provider=qa_provider,
            qa_model=qa_model,
            qa_delay=qa_delay,
            dataset_path=dataset_path,
        )
    )


async def _harden_dataset_async(
    *,
    source_name: str,
    target_name: str,
    qa_provider: str | None,
    qa_model: str | None,
    qa_delay: float,
    max_overlap: float,
) -> artifacts.DatasetInfo:
    from src.providers import llm_config_from_env
    from src.question_paraphraser import paraphrase_question_async

    qa_label = _apply_qa_model_env(qa_provider, qa_model)
    config = llm_config_from_env(provider=qa_provider, model=qa_model)

    items = artifacts.load_dataset_items(source_name)
    total = sum(len(item.get("qa_pairs", [])) for item in items)
    print(f"[INFO] Paraphrasing {total} questions from '{source_name}' with {qa_label}...")

    done = 0
    accepted = 0
    overlaps_before: list[float] = []
    overlaps_after: list[float] = []
    hardened_items = []
    for item in items:
        hardened_pairs = []
        for qa in item.get("qa_pairs", []):
            answer = qa.get("answer", "")
            answer_sentence = qa.get("answer_sentence", qa.get("answer", ""))
            result = await paraphrase_question_async(
                qa["question"],
                answer,
                answer_sentence,
                config,
                max_overlap=max_overlap,
            )
            done += 1
            accepted += int(result["accepted"])
            overlaps_before.append(result["original_overlap"])
            overlaps_after.append(result["overlap"])
            hardened_pairs.append(
                {
                    **qa,
                    "question": result["question"],
                    "question_original": qa["question"],
                    "paraphrase_overlap": round(result["overlap"], 4),
                }
            )
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}] accepted {accepted}, "
                      f"overlap {overlaps_before[-1]:.2f} -> {overlaps_after[-1]:.2f}")
            await asyncio.sleep(max(0.0, qa_delay))
        hardened_items.append({**item, "qa_pairs": hardened_pairs})

    mean_before = sum(overlaps_before) / max(1, len(overlaps_before))
    mean_after = sum(overlaps_after) / max(1, len(overlaps_after))
    print(f"[OK] Paraphrased {done} questions: accepted {accepted}/{done} "
          f"(overlap <= {max_overlap}), mean overlap {mean_before:.3f} -> {mean_after:.3f}")

    info = artifacts.save_dataset(
        target_name, hardened_items, source=f"hardened:{source_name}", qa_model=qa_label
    )
    print(f"\n[OK] Hardened dataset artifact saved: {info.label()}")
    print(f"     {artifacts.dataset_dir(target_name)}")
    return info


def harden_dataset(
    *,
    source_name: str,
    target_name: str,
    qa_provider: str | None = None,
    qa_model: str | None = None,
    qa_delay: float = 1.1,
    max_overlap: float = 0.35,
) -> artifacts.DatasetInfo:
    """Build a paraphrased ("hard") copy of an existing dataset artifact.

    Questions are rewritten away from the vocabulary of their answer sentence
    (see src/question_paraphraser.py); answers and answer sentences stay
    intact, so retrieval ground truth is unchanged.
    """
    return asyncio.run(
        _harden_dataset_async(
            source_name=source_name,
            target_name=target_name,
            qa_provider=qa_provider,
            qa_model=qa_model,
            qa_delay=qa_delay,
            max_overlap=max_overlap,
        )
    )


def prepare_embedding(
    *,
    name: str,
    provider: str,
    model: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    warm: bool = True,
) -> artifacts.EmbeddingInfo:
    """Register an embedding model (and optionally download/warm its weights)."""
    config = embedding_config_from_env(
        provider=provider, model=model, api_key_env=api_key_env, base_url=base_url
    )
    info = artifacts.EmbeddingInfo(
        name=artifacts.slugify(name),
        provider=config.provider,
        model=config.model,
        api_key_env=config.api_key_env or DEFAULT_API_KEY_ENVS.get(config.provider),
        base_url=config.base_url,
    )

    if warm:
        print(f"[INFO] Warming {config.provider}/{config.model} (downloads weights if needed)...")
        embed_model = build_embedding_model(config)
        dim = len(embed_model.get_text_embedding("test"))
        print(f"[OK] Ready (embedding dim: {dim})")

    artifacts.register_embedding(info)
    print(f"[OK] Embedding registered: {info.label()}")
    return info


def extrahard_dataset(
    *,
    source_name: str,
    target_name: str,
    partners_per_question: int = 2,
    pair_seed: int = 42,
) -> artifacts.DatasetInfo:
    """Build cross-document compound questions from an existing QA dataset."""
    from .corpus_filter import drop_unchunkable_items
    from .extrahard_pairs import build_cross_document_pairs

    corpus_items = artifacts.load_dataset_items(source_name)
    chunkable = drop_unchunkable_items(corpus_items, verbose=True)
    n_dropped = len(corpus_items) - len(chunkable)
    if n_dropped:
        print(f"[INFO] excluded {n_dropped} unchunkable doc(s) from pairing corpus")

    print(
        f"[INFO] Building cross-document pairs from '{source_name}' "
        f"({partners_per_question} partners/question, seed={pair_seed})..."
    )
    pairs = build_cross_document_pairs(
        chunkable,
        partners_per_question=partners_per_question,
        seed=pair_seed,
    )
    if not pairs:
        raise ValueError("No valid cross-document pairs were produced")

    valid_doc_ids = {str(item.get("id", idx)) for idx, item in enumerate(chunkable)}
    pairs = [
        p
        for p in pairs
        if p["source_docs"][0] in valid_doc_ids and p["source_docs"][1] in valid_doc_ids
    ]

    info = artifacts.save_extrahard_dataset(
        target_name,
        pairs,
        source_name=source_name,
        corpus_dataset=source_name,
        corpus_num_items=len(chunkable),
        partners_per_question=partners_per_question,
        pair_seed=pair_seed,
    )
    print(f"[OK] Built {len(pairs)} compound questions from {len(chunkable)} docs")
    print(f"\n[OK] Extrahard dataset artifact saved: {info.label()}")
    print(f"     {artifacts.dataset_dir(target_name)}")
    return info
