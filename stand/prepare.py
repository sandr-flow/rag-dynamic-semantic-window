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
    num_articles: int,
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
    else:
        qa_label = _apply_qa_model_env(qa_provider, qa_model)
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
    num_articles: int = 5,
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
