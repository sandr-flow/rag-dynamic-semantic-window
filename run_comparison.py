"""A/B comparison script: Current settings vs Optuna-optimized settings.

Runs benchmark on the SAME articles with two different parameter sets
for fair comparison.

Usage:
    python run_comparison.py --num-articles 20 --min-length 6000 --num-questions 3
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import DEFAULT_EXPANSION_CONFIG, DynamicSemanticConfig, ExpansionConfig
from src.metrics import compute_all_metrics
from src.strategies import DynamicSemanticStrategy
from src.tokens import count_tokens
from src.wikipedia_loader import fetch_random_articles_batch

load_dotenv()


# Optuna balanced_v3 parameters
OPTUNA_V2_PARAMS = {
    "threshold": 0.939,
    "skip_threshold": 0.936,
    "min_window": 1,
    "max_expand": 7,
    "relevance_threshold_pct": 0.698,
    "merge_gap": 2,
}

# Baseline parameters (orig_minwin1)
ORIGINAL_PARAMS = {
    "threshold": 0.85,
    "skip_threshold": 0.85,
    "min_window": 1,
    "max_expand": 7,
    "relevance_threshold_pct": 0.70,
    "merge_gap": 2,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A/B comparison benchmark")
    parser.add_argument("--num-articles", type=int, default=20)
    parser.add_argument("--min-length", type=int, default=6000)
    parser.add_argument("--num-questions", type=int, default=3)
    return parser.parse_args()


async def fetch_articles(count: int, min_length: int):
    """Fetch Wikipedia articles."""
    return await fetch_random_articles_batch(count, min_length=min_length)


async def generate_qa_pairs(articles: list, num_questions: int):
    """Generate QA pairs for articles (rate-limited)."""
    from src.question_generator import generate_qa_pairs_async
    
    results = []
    for i, (title, text) in enumerate(articles):
        qa = await generate_qa_pairs_async(text, num_questions=num_questions)
        results.append({"title": title, "text": text, "qa_pairs": qa or []})
        if i < len(articles) - 1:
            await asyncio.sleep(1.1)  # Mistral 1 RPS
    return results


def benchmark_with_params(
    documents: list[Document],
    qa_pairs: list[dict],
    title: str,
    params: dict,
    config_name: str,
) -> dict:
    """
    Benchmark DynamicSemanticStrategy with specific parameters.
    
    Returns dict with metrics.
    """
    expansion_config = ExpansionConfig(
        threshold=params.get("threshold", DEFAULT_EXPANSION_CONFIG.threshold),
        skip_threshold=params.get("skip_threshold", DEFAULT_EXPANSION_CONFIG.skip_threshold),
        max_expand=params.get("max_expand", DEFAULT_EXPANSION_CONFIG.max_expand),
        min_window=params.get("min_window", DEFAULT_EXPANSION_CONFIG.min_window),
        relevance_threshold_pct=params.get(
            "relevance_threshold_pct",
            DEFAULT_EXPANSION_CONFIG.relevance_threshold_pct,
        ),
        merge_gap=params.get("merge_gap", DEFAULT_EXPANSION_CONFIG.merge_gap),
    )
    strategy = DynamicSemanticStrategy(
        documents,
        top_k=5,
        dynamic_config=DynamicSemanticConfig(),
        expansion_config=expansion_config,
    )
    
    results = []
    
    for qa in qa_pairs:
        question = qa["question"]
        answer = qa.get("answer_sentence", qa.get("answer", ""))
        
        nodes = strategy.retrieve(question)
        texts = [n.node.text for n in nodes]
        
        metrics = compute_all_metrics(texts, answer, k=5)
        metrics["tokens"] = count_tokens(" ".join(texts))
        results.append(metrics)
    
    return {
        "config": config_name,
        "title": title,
        "results": results,
        "num_questions": len(qa_pairs),
    }


def print_comparison(current_results: list, optuna_results: list):
    """Print side-by-side comparison."""
    
    def aggregate(results_list):
        all_metrics = []
        for r in results_list:
            all_metrics.extend(r["results"])
        if not all_metrics:
            return {}
        return {
            "tokens": np.mean([m["tokens"] for m in all_metrics]),
            "hr@5": np.mean([m["hr@5"] for m in all_metrics]),
            "mrr": np.mean([m["mrr"] for m in all_metrics]),
            "precision@5": np.mean([m["precision@5"] for m in all_metrics]),
            "ndcg@5": np.mean([m["ndcg@5"] for m in all_metrics]),
        }
    
    current_agg = aggregate(current_results)
    optuna_agg = aggregate(optuna_results)
    
    print("\n" + "=" * 75)
    print("📊 A/B COMPARISON RESULTS")
    print("=" * 75)
    print(f"{'Metric':<15} | {'Current':>12} | {'Optuna':>12} | {'Δ':>10}")
    print("-" * 75)
    
    for metric in ["tokens", "hr@5", "mrr", "precision@5", "ndcg@5"]:
        curr = current_agg.get(metric, 0)
        opt = optuna_agg.get(metric, 0)
        delta = opt - curr
        delta_str = f"{delta:+.2f}" if metric != "tokens" else f"{delta:+.0f}"
        
        # Color indicator
        if metric == "tokens":
            indicator = "🟢" if delta < 0 else "🔴" if delta > 0 else "⚪"
        else:
            indicator = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
        
        print(f"{metric:<15} | {curr:>12.2f} | {opt:>12.2f} | {delta_str:>8} {indicator}")
    
    print("=" * 75)


async def main():
    """Main comparison pipeline."""
    args = parse_args()
    
    print("="*60)
    print("A/B Comparison: ORIGINAL vs OPTUNA (v2)")
    print("="*60)
    
    # Load embedding model
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"\n📦 Loading: {model_name}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    
    start = time.time()
    
    # Step 1: Fetch articles ONCE
    print(f"\n📥 Fetching {args.num_articles} articles (min {args.min_length} chars)...")
    articles = await fetch_articles(args.num_articles, args.min_length)
    print(f"   ✅ Fetched {len(articles)} articles")
    
    # Step 2: Generate QA pairs ONCE
    print(f"\n📝 Generating {args.num_questions} QA pairs per article...")
    data = await generate_qa_pairs(articles, args.num_questions)
    data = [d for d in data if d["qa_pairs"]]
    print(f"   ✅ Generated QA for {len(data)} articles")
    
    total_questions = sum(len(d["qa_pairs"]) for d in data)
    print(f"   ✅ Total questions: {total_questions}")
    
    # Step 3: Run with ORIGINAL settings
    print("\n🔵 Running with ORIGINAL settings...")
    print(f"   threshold={ORIGINAL_PARAMS['threshold']}, min_window={ORIGINAL_PARAMS['min_window']}, max_expand={ORIGINAL_PARAMS['max_expand']}")
    
    v1_results = []
    for item in data:
        documents = [Document(text=item["text"])]
        result = benchmark_with_params(
            documents, item["qa_pairs"], item["title"], ORIGINAL_PARAMS, "original"
        )
        v1_results.append(result)
        print(f"   ✓ {item['title'][:40]}...")
    
    # Step 4: Run with OPTUNA V2 settings
    print("\n🟢 Running with OPTUNA V2 (HR-focused)...")
    print(f"   threshold={OPTUNA_V2_PARAMS['threshold']:.4f}, min_window={OPTUNA_V2_PARAMS['min_window']}, max_expand={OPTUNA_V2_PARAMS['max_expand']}")
    
    v2_results = []
    for item in data:
        documents = [Document(text=item["text"])]
        result = benchmark_with_params(
            documents, item["qa_pairs"], item["title"], OPTUNA_V2_PARAMS, "optuna_v2"
        )
        v2_results.append(result)
        print(f"   ✓ {item['title'][:40]}...")
    
    # Step 5: Print comparison
    print_comparison(v1_results, v2_results)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = results_dir / f"comparison_{timestamp}.json"
    
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "original_params": ORIGINAL_PARAMS,
            "optuna_v2_params": OPTUNA_V2_PARAMS,
            "v1_results": v1_results,
            "v2_results": v2_results,
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {comparison_path}")
    print(f"⏱️ Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
