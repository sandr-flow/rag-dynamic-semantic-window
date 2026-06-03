"""Simplified benchmark script for comparing retrieval strategies."""

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
from llama_index.embeddings.mistralai import MistralAIEmbedding

from src.config import DEFAULT_BENCHMARK_CONFIG, DEFAULT_RETRIEVAL_CONFIG
from src.metrics import compute_all_metrics
from src.strategies import (
    DynamicSemanticStrategy,
    FixedWindowStrategy,
    NaiveChunkingStrategy,
    SemanticSplitterStrategy,
)
from src.utils import load_text_file

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark retrieval strategies")
    parser.add_argument(
        "--source", 
        choices=["static", "wikipedia", "qasper"],
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
    return parser.parse_args()


def count_tokens(text: str) -> int:
    """Rough token count (chars / 4)."""
    return len(text) // 4


def benchmark_article(text: str, qa_pairs: list[dict], title: str, save_full_text: bool = True) -> dict:
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
    
    top_k = DEFAULT_RETRIEVAL_CONFIG.top_k
    
    strategies = [
        NaiveChunkingStrategy(documents, top_k=top_k),
        FixedWindowStrategy(documents, top_k=top_k),
        SemanticSplitterStrategy(documents, top_k=top_k),
        DynamicSemanticStrategy(
            documents,
            top_k=top_k,
            seed_rejection_log_path=seed_log_path,
        ),
    ]
    
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
            
            metrics = compute_all_metrics(texts, answer, k=5)
            metrics["tokens"] = count_tokens(" ".join(texts))
            results[strategy.name].append(metrics)
            
            strategy_chunks[strategy.name] = texts
            strategy_metrics[strategy.name] = metrics
        
        # Log failure: Dynamic Semantic missed but at least one other strategy hit
        dynamic_hr = strategy_metrics.get("Dynamic Semantic", {}).get("hr@5", 0)
        other_hrs = [
            strategy_metrics.get(name, {}).get("hr@5", 0)
            for name in ["Naive Chunking", "Fixed Window", "Semantic Splitter"]
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
                        "hr@5": strategy_metrics[name]["hr@5"],
                        "mrr": strategy_metrics[name]["mrr"],
                        "tokens": strategy_metrics[name]["tokens"],
                        "chunks": strategy_chunks[name],
                    }
                    for name in strategy_chunks
                },
            }
            
            with open(failures_dir / filename, "w", encoding="utf-8") as f:
                json.dump(failure_log, f, indent=2, ensure_ascii=False)
    
    return {"title": title, "results": results, "num_questions": len(qa_pairs)}


async def fetch_wikipedia_articles(count: int, specific_title: str = None, min_length: int = 2000):
    """Fetch Wikipedia articles."""
    from src.wikipedia_loader import fetch_article_by_title, fetch_random_articles_batch

    if specific_title:
        title, text = fetch_article_by_title(specific_title)
        return [(title, text)]
    return await fetch_random_articles_batch(count, min_length=min_length)


async def generate_qa_pairs(articles: list, num_questions: int):
    """Generate QA pairs for articles (rate-limited)."""
    from src.question_generator import generate_qa_pairs_async
    
    results = []
    for i, (title, text) in enumerate(articles):
        qa = await generate_qa_pairs_async(text, num_questions=num_questions)
        results.append({"title": title, "text": text, "qa_pairs": qa or []})
        if i < len(articles) - 1:
            await asyncio.sleep(DEFAULT_BENCHMARK_CONFIG.mistral_rps_delay)
    return results


def run_benchmark():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("Dynamic Semantic Window Benchmark")
    print("=" * 60)
    
    # Load embedding model
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    
    if embedding_provider == "mistral":
        print("Loading: mistral-embed")
        Settings.embed_model = MistralAIEmbedding(
            model_name="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
    else:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        print(f"Loading: {model_name}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    
    start = time.time()
    
    all_results = []
    
    # Get articles + QA pairs
    data = []
    if args.source == "wikipedia":
        print(f"\n📥 Fetching {args.num_articles} Wikipedia articles (min length: {args.min_length})...")
        articles = asyncio.run(fetch_wikipedia_articles(args.num_articles, args.article, args.min_length))
        print(f"✅ Fetched {len(articles)} articles")

        print(f"\n📝 Generating QA pairs...")
        data = asyncio.run(generate_qa_pairs(articles, args.num_questions))
        data = [d for d in data if d["qa_pairs"]]  # Filter failed
        print(f"✅ Generated QA for {len(data)} articles")

    elif args.source == "qasper":
        from src.qasper_loader import fetch_qasper_articles

        skip_msg = f", skipping first {args.skip}" if args.skip > 0 else ""
        print(f"\n📥 Loading {args.num_articles} QASPER articles (split: {args.split}, min length: {args.min_length}{skip_msg})...")
        articles = fetch_qasper_articles(args.num_articles, args.min_length, split=args.split, skip=args.skip)
        print(f"✅ Loaded {len(articles)} articles")

        print(f"\n📝 Generating QA pairs...")
        data = asyncio.run(generate_qa_pairs(articles, args.num_questions))
        data = [d for d in data if d["qa_pairs"]]  # Filter failed
        print(f"✅ Generated QA for {len(data)} articles")

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
    print(f"\n🔬 Benchmarking {len(data)} articles...")
    for item in data:
        print(f"  → {item['title']}")
        result = benchmark_article(item["text"], item["qa_pairs"], item["title"])
        all_results.append(result)
    # Aggregate
    agg = {}
    for result in all_results:
        for strategy, metrics in result["results"].items():
            if strategy not in agg:
                agg[strategy] = []
            agg[strategy].extend(metrics)
    
    # Print results
    total_q = sum(r["num_questions"] for r in all_results)
    print(f"\n📊 RESULTS ({len(all_results)} datasets/articles, {total_q} questions)")
    
    # Determine columns based on first available metrics
    first_strategy = next(iter(agg)) if agg else None
    if first_strategy and agg[first_strategy]:
        sample_metrics = agg[first_strategy][0]
        has_recall = "recall@2" in sample_metrics
        
        print("-" * 85)
        if has_recall:
             print(f"{'Strategy':20} | {'Tokens':>7} | {'R@2':>6} | {'R@5':>6} | {'NDCG@5':>7}")
        else:
            print(f"{'Strategy':20} | {'Tokens':>7} | {'HR@5':>6} | {'MRR':>6} | {'P@5':>6} | {'NDCG@5':>6}")
        print("-" * 85)
        
        for strategy, metrics in agg.items():
            if not metrics:
                continue
            avg_tokens = np.mean([m["tokens"] for m in metrics])
            
            if has_recall:
                avg_r2 = np.mean([m["recall@2"] for m in metrics])
                avg_r5 = np.mean([m["recall@5"] for m in metrics])
                avg_ndcg = np.mean([m["ndcg@5"] for m in metrics])
                print(f"{strategy:20} | {avg_tokens:7.1f} | {avg_r2:6.4f} | {avg_r5:6.4f} | {avg_ndcg:7.4f}")
            else:
                avg_hr = np.mean([m["hr@5"] for m in metrics])
                avg_mrr = np.mean([m["mrr"] for m in metrics])
                avg_p = np.mean([m["precision@5"] for m in metrics])
                avg_ndcg = np.mean([m["ndcg@5"] for m in metrics])
                print(f"{strategy:20} | {avg_tokens:7.1f} | {avg_hr:6.4f} | {avg_mrr:6.4f} | {avg_p:6.4f} | {avg_ndcg:6.4f}")
    
    # Save
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(results_dir / f"benchmark_{timestamp}.json", "w") as f:
        json.dump({"config": vars(args), "results": all_results, "aggregate": {k: [dict(m) for m in v] for k, v in agg.items()}}, f, indent=2)
    
    print(f"\n⏱️ Total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    run_benchmark()

