"""Simplified benchmark script for comparing retrieval strategies."""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Document, Settings

from src.benchmark_datasets import load_benchmark_dataset
from src.config import DEFAULT_BENCHMARK_CONFIG, DEFAULT_RETRIEVAL_CONFIG
from src.metrics import compute_all_metrics
from src.providers import (
    build_embedding_model,
    embedding_config_from_env,
    llm_config_from_env,
    provider_catalog,
)
from src.strategy_registry import (
    available_strategy_ids,
    create_strategy,
    parse_strategy_ids,
    resolve_strategy_overrides,
    strategy_catalog,
    validate_strategy_overrides_for_ids,
)
from src.utils import load_text_file

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark retrieval strategies")
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print supported strategy ids and exit.",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="Print supported embedding/LLM providers and exit.",
    )
    parser.add_argument(
        "--source", 
        choices=["static", "wikipedia", "qasper", "custom"],
        default="static"
    )
    parser.add_argument(
        "--num-questions", 
        type=int, 
        default=DEFAULT_BENCHMARK_CONFIG.default_num_questions,
    )
    parser.add_argument(
        "--num-articles", 
        type=int, 
        default=DEFAULT_BENCHMARK_CONFIG.default_num_articles,
    )
    parser.add_argument("--article", type=str, default=None)
    parser.add_argument(
        "--min-length", 
        type=int, 
        default=DEFAULT_BENCHMARK_CONFIG.default_min_article_length,
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of articles to skip (for non-overlapping validation sets)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="QASPER dataset split (train has ~1500 articles)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="default",
        help=f"Comma-separated strategy ids. Available: {', '.join(available_strategy_ids())}",
    )
    parser.add_argument(
        "--strategy-overrides-json",
        type=str,
        default=None,
        help="Path to JSON with strategy parameter overrides, e.g. results/best_params.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", DEFAULT_RETRIEVAL_CONFIG.top_k)),
        help="Number of chunks/clusters to retrieve per strategy",
    )
    parser.add_argument(
        "--metric-k",
        type=int,
        default=None,
        help="Cutoff for HR/P/R/NDCG metrics. Defaults to --top-k.",
    )
    parser.add_argument("--embedding-provider", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--embedding-api-key-env", type=str, default=None)
    parser.add_argument("--embedding-base-url", type=str, default=None)
    parser.add_argument("--llm-provider", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-api-key-env", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument(
        "--qa-delay",
        type=float,
        default=float(os.getenv("QA_GENERATION_DELAY", DEFAULT_BENCHMARK_CONFIG.qa_generation_delay)),
        help="Delay between QA-generation LLM calls in seconds.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Combined JSON/JSONL benchmark dataset with text and qa_pairs.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Human-readable dataset label stored in benchmark result metadata.",
    )
    parser.add_argument(
        "--articles-path",
        type=str,
        default=None,
        help="Articles JSON/JSONL path for paired custom datasets.",
    )
    parser.add_argument(
        "--questions-path",
        type=str,
        default=None,
        help="Questions JSON/JSONL path for paired custom datasets.",
    )
    return parser.parse_args()


def count_tokens(text: str) -> int:
    """Rough token count (chars / 4)."""
    return len(text) // 4


def load_strategy_overrides(path: str | None) -> dict:
    """Load strategy overrides from JSON or Optuna best_params artifact."""
    if not path:
        return {}
    with open(path, encoding="utf-8-sig") as f:
        payload = json.load(f)
    if "params" in payload and isinstance(payload["params"], dict):
        return payload["params"]
    return payload


def benchmark_article(
    text: str,
    qa_pairs: list[dict],
    title: str,
    strategy_ids: list[str] | None = None,
    top_k: int = DEFAULT_RETRIEVAL_CONFIG.top_k,
    metric_k: int | None = None,
    strategy_overrides: dict | None = None,
    save_full_text: bool = True,
) -> dict:
    """
    Benchmark all strategies on a single article.
    
    Returns dict with metrics per strategy.
    """
    documents = [Document(text=text)]
    
    # Seed rejection log path
    results_dir = Path(__file__).parent / "results"
    seed_log_path = str(results_dir / "seed_rejections.jsonl")
    failures_dir = results_dir / "failures"
    failures_dir.mkdir(exist_ok=True)
    
    strategy_ids = strategy_ids or parse_strategy_ids("default")
    metric_k = metric_k or top_k
    strategies = []
    strategy_id_by_name = {}
    effective_strategy_overrides = {}
    for strategy_id in strategy_ids:
        effective_overrides = resolve_strategy_overrides(strategy_id, strategy_overrides)
        strategy = create_strategy(
            strategy_id,
            documents,
            top_k=top_k,
            seed_rejection_log_path=seed_log_path,
            overrides=effective_overrides,
        )
        strategies.append(strategy)
        strategy_id_by_name[strategy.name] = strategy_id
        effective_strategy_overrides[strategy.name] = effective_overrides
    
    results = {s.name: [] for s in strategies}
    
    for qa in qa_pairs:
        question = qa["question"]
        answer = qa.get("answer_sentence", qa.get("answer", ""))
        
        # Store chunks for failure logging
        strategy_chunks = {}
        strategy_metrics = {}
        
        for strategy in strategies:
            nodes = strategy.retrieve(question)
            texts = [n.node.text for n in nodes]
            
            metrics = compute_all_metrics(texts, answer, k=metric_k)
            metrics["tokens"] = count_tokens(" ".join(texts))
            results[strategy.name].append(metrics)
            
            strategy_chunks[strategy.name] = texts
            strategy_metrics[strategy.name] = metrics
        
        # Log failure: Dynamic Semantic missed but at least one other strategy hit
        hr_key = f"hr@{metric_k}"
        dynamic_hr = strategy_metrics.get("Dynamic Semantic", {}).get(hr_key, 0)
        other_hrs = [
            metrics.get(hr_key, 0)
            for name, metrics in strategy_metrics.items()
            if name != "Dynamic Semantic"
        ]
        
        if dynamic_hr == 0 and any(hr > 0 for hr in other_hrs):
            # Generate safe filename
            safe_title = "".join(c if c.isalnum() else "_" for c in title)[:50]
            safe_question = "".join(c if c.isalnum() else "_" for c in question)[:30]
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{safe_title}_{safe_question}_{timestamp}.json"
            
            failure_log = {
                "article_title": title,
                "article_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "full_text": text if save_full_text else None,  # Full article for Optuna corpus
                "all_qa_pairs": qa_pairs if save_full_text else None,  # All QA pairs for corpus
                "failed_question_index": qa_pairs.index(qa),  # Which question failed
                "question": question,
                "expected_answer": answer,
                "strategies": {
                    name: {
                        hr_key: strategy_metrics[name][hr_key],
                        "mrr": strategy_metrics[name]["mrr"],
                        "tokens": strategy_metrics[name]["tokens"],
                        "chunks": strategy_chunks[name],
                    }
                    for name in strategy_chunks
                },
            }
            
            with open(failures_dir / filename, "w", encoding="utf-8") as f:
                json.dump(failure_log, f, indent=2, ensure_ascii=False)
    
    return {
        "title": title,
        "results": results,
        "num_questions": len(qa_pairs),
        "strategy_id_by_name": strategy_id_by_name,
        "effective_strategy_overrides": effective_strategy_overrides,
    }


async def fetch_wikipedia_articles(count: int, specific_title: str = None, min_length: int = 2000):
    """Fetch Wikipedia articles."""
    from src.wikipedia_loader import fetch_article_by_title, fetch_random_articles_batch

    if specific_title:
        title, text = fetch_article_by_title(specific_title)
        return [(title, text)]
    return await fetch_random_articles_batch(count, min_length=min_length)


async def generate_qa_pairs(articles: list, num_questions: int, qa_delay: float):
    """Generate QA pairs for articles (rate-limited)."""
    from src.question_generator import generate_qa_pairs_async
    
    results = []
    for i, (title, text) in enumerate(articles):
        qa = await generate_qa_pairs_async(text, num_questions=num_questions)
        results.append({"title": title, "text": text, "qa_pairs": qa or []})
        if i < len(articles) - 1:
            await asyncio.sleep(max(0.0, qa_delay))
    return results


def run_benchmark():
    """Main entry point."""
    args = parse_args()

    if args.list_strategies:
        print_strategy_catalog()
        return
    if args.list_providers:
        print_provider_catalog()
        return
    
    print("=" * 60)
    print("Dynamic Semantic Window Benchmark")
    print("=" * 60)
    
    # Make LLM CLI overrides visible to question_generator.
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.llm_api_key_env:
        os.environ["LLM_API_KEY_ENV"] = args.llm_api_key_env
    if args.llm_base_url:
        os.environ["LLM_BASE_URL"] = args.llm_base_url

    strategy_ids = parse_strategy_ids(args.strategies)
    strategy_overrides = load_strategy_overrides(args.strategy_overrides_json)
    override_errors = validate_strategy_overrides_for_ids(strategy_ids, strategy_overrides)
    if override_errors:
        raise ValueError("Invalid strategy overrides:\n" + "\n".join(f"- {error}" for error in override_errors))
    metric_k = args.metric_k or args.top_k

    # Load embedding model
    embedding_config = embedding_config_from_env(
        provider=args.embedding_provider,
        model=args.embedding_model,
        api_key_env=args.embedding_api_key_env,
        base_url=args.embedding_base_url,
    )
    llm_used_for_qa_generation = args.source in {"wikipedia", "qasper"}
    llm_config = None
    if llm_used_for_qa_generation or any(
        [args.llm_provider, args.llm_model, args.llm_api_key_env, args.llm_base_url]
    ):
        llm_config = llm_config_from_env(
            provider=args.llm_provider,
            model=args.llm_model,
            api_key_env=args.llm_api_key_env,
            base_url=args.llm_base_url,
        )
    print(f"Embedding: {embedding_config.provider}/{embedding_config.model}")
    if llm_config:
        usage = "QA generation" if llm_used_for_qa_generation else "configured"
        print(f"LLM ({usage}): {llm_config.provider}/{llm_config.model}")
    Settings.embed_model = build_embedding_model(embedding_config)
    
    start = time.time()
    
    all_results = []
    
    # Get articles + QA pairs
    data = []
    if args.source == "custom" or args.dataset_path or (args.articles_path and args.questions_path):
        print("\nLoading custom benchmark dataset...")
        data = load_benchmark_dataset(
            dataset_path=args.dataset_path,
            articles_path=args.articles_path,
            questions_path=args.questions_path,
        )
        if args.num_articles:
            data = data[: args.num_articles]
        print(f"Loaded {len(data)} custom benchmark items")

    elif args.source == "wikipedia":
        print(f"\n[INFO] Fetching {args.num_articles} Wikipedia articles (min length: {args.min_length})...")
        articles = asyncio.run(fetch_wikipedia_articles(args.num_articles, args.article, args.min_length))
        print(f"[OK] Fetched {len(articles)} articles")

        print("\n[INFO] Generating QA pairs...")
        data = asyncio.run(generate_qa_pairs(articles, args.num_questions, args.qa_delay))
        data = [d for d in data if d["qa_pairs"]]  # Filter failed
        print(f"[OK] Generated QA for {len(data)} articles")

    elif args.source == "qasper":
        from src.qasper_loader import fetch_qasper_articles

        skip_msg = f", skipping first {args.skip}" if args.skip > 0 else ""
        print(f"\n[INFO] Loading {args.num_articles} QASPER articles (split: {args.split}, min length: {args.min_length}{skip_msg})...")
        articles = fetch_qasper_articles(args.num_articles, args.min_length, split=args.split, skip=args.skip)
        print(f"[OK] Loaded {len(articles)} articles")

        print("\n[INFO] Generating QA pairs...")
        data = asyncio.run(generate_qa_pairs(articles, args.num_questions, args.qa_delay))
        data = [d for d in data if d["qa_pairs"]]  # Filter failed
        print(f"[OK] Generated QA for {len(data)} articles")

    else:
        # Static mode
        text_path = Path(__file__).parent / "data" / "source_text.txt"
        text = load_text_file(str(text_path))
        data = [{
            "title": "Static",
            "text": text,
            "qa_pairs": [
                {"question": "What is quantum superposition?", "answer_sentence": "Quantum superposition is the principle that a quantum system can exist in multiple states simultaneously until measured."},
                {"question": "What is quantum entanglement?", "answer_sentence": "Quantum entanglement occurs when particles become correlated such that the quantum state of one particle cannot be described independently."},
                {"question": "What is the Heisenberg uncertainty principle?", "answer_sentence": "The Heisenberg uncertainty principle states that one cannot simultaneously know both the exact position and momentum of a particle."},
            ]
        }]

    # Run benchmarks
    print(f"\n[INFO] Benchmarking {len(data)} articles...")
    print(f"Strategies: {', '.join(strategy_ids)}")
    print(f"Top-K: {args.top_k}, Metric-K: {metric_k}")
    if strategy_overrides:
        print(f"Strategy overrides: {strategy_overrides}")
    for item in data:
        print(f"  -> {item['title']}")
        result = benchmark_article(
            item["text"],
            item["qa_pairs"],
            item["title"],
            strategy_ids=strategy_ids,
            top_k=args.top_k,
            metric_k=metric_k,
            strategy_overrides=strategy_overrides,
        )
        all_results.append(result)
    # Aggregate
    agg = {}
    strategy_id_by_name = {}
    effective_strategy_overrides = {}
    for result in all_results:
        strategy_id_by_name.update(result.get("strategy_id_by_name") or {})
        effective_strategy_overrides.update(result.get("effective_strategy_overrides") or {})
        for strategy, metrics in result["results"].items():
            if strategy not in agg:
                agg[strategy] = []
            agg[strategy].extend(metrics)
    
    # Print results
    total_q = sum(r["num_questions"] for r in all_results)
    print(f"\nRESULTS ({len(all_results)} datasets/articles, {total_q} questions)")
    
    # Determine columns based on first available metrics
    first_strategy = next(iter(agg)) if agg else None
    if first_strategy and agg[first_strategy]:
        metric_suffix = f"@{metric_k}"
        
        print("-" * 85)
        print(
            f"{'Strategy':20} | {'Tokens':>7} | {f'HR{metric_suffix}':>6} | "
            f"{'MRR':>6} | {f'P{metric_suffix}':>6} | {f'NDCG{metric_suffix}':>8}"
        )
        print("-" * 85)
        
        for strategy, metrics in agg.items():
            if not metrics:
                continue
            avg_tokens = np.mean([m["tokens"] for m in metrics])
            avg_hr = np.mean([m[f"hr@{metric_k}"] for m in metrics])
            avg_mrr = np.mean([m["mrr"] for m in metrics])
            avg_p = np.mean([m[f"precision@{metric_k}"] for m in metrics])
            avg_ndcg = np.mean([m[f"ndcg@{metric_k}"] for m in metrics])
            print(
                f"{strategy:20} | {avg_tokens:7.1f} | {avg_hr:6.4f} | "
                f"{avg_mrr:6.4f} | {avg_p:6.4f} | {avg_ndcg:8.4f}"
            )
    
    # Save
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_config = vars(args).copy()
    result_config["metric_k"] = metric_k
    result_config["actual_num_articles"] = len(all_results)
    result_config["actual_num_questions"] = total_q
    result_config["requested_num_articles"] = args.num_articles
    result_config["questions_per_article"] = args.num_questions
    with open(results_dir / f"benchmark_{timestamp}.json", "w") as f:
        json.dump(
            {
                "config": result_config,
                "embedding": {
                    "provider": embedding_config.provider,
                    "model": embedding_config.model,
                    "base_url": embedding_config.base_url,
                },
                "llm": {
                    "provider": llm_config.provider if llm_config else None,
                    "model": llm_config.model if llm_config else None,
                    "base_url": llm_config.base_url if llm_config else None,
                    "used_for_qa_generation": llm_used_for_qa_generation,
                },
                "strategies": strategy_ids,
                "strategy_overrides": strategy_overrides,
                "strategy_id_by_name": strategy_id_by_name,
                "effective_strategy_overrides": effective_strategy_overrides,
                "results": all_results,
                "aggregate": {k: [dict(m) for m in v] for k, v in agg.items()},
            },
            f,
            indent=2,
        )
    
    print(f"\nTotal: {time.time() - start:.1f}s")


def print_strategy_catalog() -> None:
    """Print supported retrieval strategy ids."""
    print("Strategies")
    print("=" * 60)
    for item in strategy_catalog():
        marker = "default" if item["default"] else "optional"
        overrides = ", ".join(item["override_keys"])
        print(f"{item['id']:18} {marker:8} {item['description']}")
        print(f"{'':18} overrides: {overrides}")
        print(f"{'':18} defaults: {json.dumps(item['default_params'], sort_keys=True)}")


def print_provider_catalog() -> None:
    """Print supported embedding and LLM providers."""
    catalog = provider_catalog()
    print("Embedding providers")
    print("=" * 60)
    for item in catalog["embeddings"]:
        key = item["api_key_env"] or "-"
        base_url = item["base_url"] or "-"
        print(
            f"{item['id']:12} default_model={item['default_model']} "
            f"api_key_env={key} base_url={base_url}"
        )

    print("\nLLM providers")
    print("=" * 60)
    for item in catalog["llms"]:
        key = item["api_key_env"] or "-"
        base_url = item["base_url"] or "-"
        print(
            f"{item['id']:12} default_model={item['default_model']} "
            f"api_key_env={key} base_url={base_url}"
        )


def main() -> int:
    """CLI entrypoint with concise configuration errors."""
    try:
        run_benchmark()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

