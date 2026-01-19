"""Benchmark script for comparing retrieval strategies."""

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode

from src.metrics import compute_all_metrics
from src.strategies import (
    DynamicSemanticStrategy,
    FixedWindowStrategy,
    NaiveChunkingStrategy,
    SemanticSplitterStrategy,
)
from src.utils import load_text_file

# Load environment variables
load_dotenv()

# Pre-load NLTK to prevent thread-safety issues
import nltk
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# Force load the lazy loader
from nltk.tokenize import sent_tokenize
try:
    sent_tokenize("Dummy sentence to force load.")
except Exception:
    pass

# Global Embedding Cache
EMBEDDING_CACHE: Dict[str, List[float]] = {}

# Default QA pairs for static mode
DEFAULT_QA_PAIRS = [
    {"question": "What is quantum superposition?", "answer": "multiple states simultaneously", "answer_sentence": "Quantum superposition is the principle that a quantum system can exist in multiple states simultaneously until measured."},
    {"question": "How does quantum entanglement work?", "answer": "correlated states", "answer_sentence": "Quantum entanglement occurs when particles become correlated such that the quantum state of one particle cannot be described independently."},
    {"question": "What is the Heisenberg uncertainty principle?", "answer": "cannot precisely measure both position and momentum", "answer_sentence": "The Heisenberg uncertainty principle states that one cannot simultaneously know both the exact position and momentum of a particle."},
    {"question": "What is quantum tunneling?", "answer": "passing through barriers", "answer_sentence": "Quantum tunneling is the phenomenon where particles pass through potential barriers that would be insurmountable in classical physics."},
    {"question": "What is the Copenhagen interpretation?", "answer": "wave function collapse upon measurement", "answer_sentence": "The Copenhagen interpretation posits that quantum systems exist in superposition until observed, at which point the wave function collapses."},
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark retrieval strategies for RAG")
    parser.add_argument("--source", choices=["static", "wikipedia"], default="static", help="Data source")
    parser.add_argument("--num-questions", type=int, default=5, help="Questions per article")
    parser.add_argument("--num-articles", type=int, default=1, help="Number of articles (Wikipedia mode)")
    parser.add_argument("--article", type=str, default=None, help="Specific article title")
    return parser.parse_args()


def get_cached_embedding(text: str) -> List[float]:
    """Get embedding from cache or compute it."""
    if text not in EMBEDDING_CACHE:
        EMBEDDING_CACHE[text] = Settings.embed_model.get_text_embedding(text)
    return EMBEDDING_CACHE[text]


def count_tokens(text: str) -> int:
    return len(text) // 4


def calculate_coherence(nodes: list) -> float:
    """Calculate average intra-cluster coherence using cached embeddings."""
    if len(nodes) < 2:
        return 1.0

    similarities = []
    for i in range(len(nodes) - 1):
        # Use node.embedding if available, else cache
        emb1 = nodes[i].node.embedding or get_cached_embedding(nodes[i].node.text)
        emb2 = nodes[i + 1].node.embedding or get_cached_embedding(nodes[i + 1].node.text)
        
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarities.append(sim)

    return float(np.mean(similarities))


def benchmark_single_article(
    text: str,
    qa_pairs: list[dict],
    article_title: str,
    strategies_config: dict
) -> dict:
    """
    Run benchmark on a single article. 
    Intended to be run in a thread.
    """
    try:
        print(f"  🏁 Starting benchmark for: {article_title}")
        
        # Initialize strategies locally for this thread/article
        documents = [Document(text=text)]
        strategies = [
            NaiveChunkingStrategy(documents, top_k=5),
            FixedWindowStrategy(documents, top_k=5),
            SemanticSplitterStrategy(documents, top_k=5),
            DynamicSemanticStrategy(
                documents,
                top_k=5,
                threshold=strategies_config["threshold"],
                max_expand=strategies_config["max_expand"],
            ),
        ]

        article_metrics = {s.name: [] for s in strategies}

        for qa in qa_pairs:
            for strategy in strategies:
                nodes = strategy.retrieve(qa["question"])
                retrieved_texts = [n.node.text for n in nodes]
                answer_sentence = qa.get("answer_sentence", qa.get("answer", ""))

                ir_metrics = compute_all_metrics(retrieved_texts, answer_sentence, k=5)
                token_count = count_tokens(" ".join(retrieved_texts))
                coherence = calculate_coherence(nodes)

                article_metrics[strategy.name].append({
                    "token_count": token_count,
                    "coherence": coherence,
                    **ir_metrics,
                })

        # Calculate averages for this article
        summary = {}
        for s_name, metrics in article_metrics.items():
            if metrics:
                summary[s_name] = {
                    "avg_hr@5": np.mean([m["hr@5"] for m in metrics]),
                    "avg_mrr": np.mean([m["mrr"] for m in metrics]),
                    "metrics_list": metrics # Keep full list for aggregation
                }
        
        print(f"  ✅ Finished benchmark for: {article_title}")
        return {
            "title": article_title,
            "url": f"https://en.wikipedia.org/wiki/{article_title.replace(' ', '_')}",
            "num_questions": len(qa_pairs),
            "results": summary, 
            "raw_metrics": article_metrics
        }

    except Exception as e:
        print(f"  ❌ Error benchmarking {article_title}: {e}")
        return None


async def process_wikipedia_mode(args):
    """Async orchestration for Wikipedia mode."""
    from src.question_generator import generate_qa_pairs_async
    from src.wikipedia_loader import fetch_article_by_title, fetch_random_article

    tasks = []
    num_articles = 1 if args.article else args.num_articles
    
    # 1. Fetch and Generate QA Pairs (Buffered/Rate Limited)
    print(f"\n🚀 Phase 1: Fetching {num_articles} articles and generating QA pairs (Async)...")
    
    article_data = []

    for i in range(num_articles):
        # Fetch Text (Sync, run in thread to not block)
        if args.article:
            title, text = await asyncio.to_thread(fetch_article_by_title, args.article)
        else:
            title, text = await asyncio.to_thread(fetch_random_article)

        print(f"  📄 [{i+1}/{num_articles}] Fetched: {title} ({len(text)} chars)")
        
        # Start async QA generation
        # We process strictly sequentially regarding the START time to respect 1 RPS
        # But we await the result so we don't spam requests.
        # Actually proper 1 RPS means we can fire one every 1s. 
        # But for simplicity and safety, we'll do:
        # Request -> Await -> Sleep 1.1s. 
        # This is safe but slower. 
        # Fast way: Fire Request, Sleep 1.1s, Fire Request (don't await yet).
        
        task = asyncio.create_task(generate_qa_pairs_async(text, num_questions=args.num_questions))
        article_data.append({"title": title, "text": text, "task": task})
        
        if i < num_articles - 1:
            await asyncio.sleep(1.1)  # Rate limit safety

    # Wait for all QA tasks to finish
    print("  ⏳ Waiting for QA generation to complete...")
    for item in article_data:
        item["qa_pairs"] = await item["task"]
        if not item["qa_pairs"]:
            print(f"  ⚠️ No QA pairs for {item['title']}")
    
    valid_articles = [a for a in article_data if a.get("qa_pairs")]
    print(f"  ✅ Ready to benchmark {len(valid_articles)} articles.")

    # 2. Parallel Benchmarking
    print(f"\n🚀 Phase 2: Benchmarking articles in parallel threads...")
    
    strategies_config = {
        "threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.6")),
        "max_expand": int(os.getenv("MAX_EXPAND", "5")),
    }

    results = []
    
    # Run CPU/Embedding heavy benchmark in threads
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
        futures = {
            executor.submit(
                benchmark_single_article, 
                a["text"], 
                a["qa_pairs"], 
                a["title"], 
                strategies_config
            ): a for a in valid_articles
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    return results


def run_benchmark():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("Dynamic Semantic Window Benchmark (Optimized)")
    print("=" * 60)

    # Load model once
    embed_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"Loading embedding model: {embed_model_name}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    start_time = time.time()
    
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "articles": [],
        "aggregate_metrics": {}
    }

    processed_articles = []

    if args.source == "wikipedia":
        processed_articles = asyncio.run(process_wikipedia_mode(args))
    else:
        # Static Mode
        data_path = Path(__file__).parent / "data" / "source_text.txt"
        print(f"Loading test data from: {data_path}")
        text = load_text_file(str(data_path))
        
        strategies_config = {
            "threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.6")),
            "max_expand": int(os.getenv("MAX_EXPAND", "5")),
        }
        
        res = benchmark_single_article(text, DEFAULT_QA_PAIRS, "Static Text", strategies_config)
        if res:
            processed_articles.append(res)

    # Aggregation
    all_raw_metrics = {} # strategy -> list of all metrics dicts

    for art_res in processed_articles:
        final_results["articles"].append({
            "title": art_res["title"],
            "url": art_res["url"],
            "num_questions": art_res["num_questions"]
        })
        
        for strat, metrics_list in art_res["raw_metrics"].items():
            if strat not in all_raw_metrics:
                all_raw_metrics[strat] = []
            all_raw_metrics[strat].extend(metrics_list)

    # Compute Aggregates
    print(f"\n📊 AGGREGATE RESULTS ({len(processed_articles)} articles, {len(all_raw_metrics.get('DynamicSemanticStrategy', []))} questions):")
    print("-" * 90)
    print(f"{'Strategy':20} | {'Tokens':>7} | {'Coher':>5} | {'HR@5':>5} | {'MRR':>5} | {'P@5':>5} | {'NDCG':>5}")
    print("-" * 90)

    for strat, metrics in all_raw_metrics.items():
        if not metrics:
            continue
            
        agg = {
            "avg_tokens": float(np.mean([m["token_count"] for m in metrics])),
            "avg_coherence": float(np.mean([m["coherence"] for m in metrics])),
            "avg_hr@5": float(np.mean([m["hr@5"] for m in metrics])),
            "avg_mrr": float(np.mean([m["mrr"] for m in metrics])),
            "avg_precision@5": float(np.mean([m["precision@5"] for m in metrics])),
            "avg_ndcg@5": float(np.mean([m["ndcg@5"] for m in metrics])),
        }
        final_results["aggregate_metrics"][strat] = agg
        
        print(f"{strat:20} | {agg['avg_tokens']:7.1f} | {agg['avg_coherence']:5.3f} | "
              f"{agg['avg_hr@5']:5.2f} | {agg['avg_mrr']:5.2f} | {agg['avg_precision@5']:5.2f} | {agg['avg_ndcg@5']:5.2f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).parent / "results" / f"benchmark_{timestamp}.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n⏱️ Total time: {time.time() - start_time:.2f}s")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
