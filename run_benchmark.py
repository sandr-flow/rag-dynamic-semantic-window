"""Benchmark script for comparing retrieval strategies."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.metrics import compute_all_metrics
from src.strategies import (
    DynamicSemanticStrategy,
    FixedWindowStrategy,
    NaiveChunkingStrategy,
)
from src.utils import load_text_file

# Load environment variables
load_dotenv()

# Default QA pairs for static mode (quantum mechanics)
DEFAULT_QA_PAIRS = [
    {"question": "What is quantum superposition?", "answer": "multiple states simultaneously", "answer_sentence": "Quantum superposition is the principle that a quantum system can exist in multiple states simultaneously until measured."},
    {"question": "How does quantum entanglement work?", "answer": "correlated states", "answer_sentence": "Quantum entanglement occurs when particles become correlated such that the quantum state of one particle cannot be described independently."},
    {"question": "What is the Heisenberg uncertainty principle?", "answer": "cannot precisely measure both position and momentum", "answer_sentence": "The Heisenberg uncertainty principle states that one cannot simultaneously know both the exact position and momentum of a particle."},
    {"question": "What is quantum tunneling?", "answer": "passing through barriers", "answer_sentence": "Quantum tunneling is the phenomenon where particles pass through potential barriers that would be insurmountable in classical physics."},
    {"question": "What is the Copenhagen interpretation?", "answer": "wave function collapse upon measurement", "answer_sentence": "The Copenhagen interpretation posits that quantum systems exist in superposition until observed, at which point the wave function collapses."},
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval strategies for RAG"
    )
    parser.add_argument(
        "--source",
        choices=["static", "wikipedia"],
        default="static",
        help="Data source: static file or random Wikipedia article",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions per article",
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=1,
        help="Number of random Wikipedia articles to benchmark (Wikipedia mode)",
    )
    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help="Specific Wikipedia article title (optional, overrides --num-articles)",
    )
    return parser.parse_args()


def count_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 4 chars per token)."""
    return len(text) // 4


def calculate_coherence(nodes: list) -> float:
    """
    Calculate average intra-cluster coherence.

    Returns mean similarity between consecutive chunks.
    """
    if len(nodes) < 2:
        return 1.0

    similarities = []
    embed_model = Settings.embed_model

    for i in range(len(nodes) - 1):
        emb1 = embed_model.get_text_embedding(nodes[i].node.text)
        emb2 = embed_model.get_text_embedding(nodes[i + 1].node.text)
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarities.append(sim)

    return float(np.mean(similarities))


def benchmark_single_article(
    text: str,
    qa_pairs: list[dict],
    strategies: list,
    article_name: str,
    logs_dir: Path,
) -> dict:
    """
    Run benchmark on a single article.
    
    Returns dict with article results and metrics.
    """
    article_result = {
        "article": article_name,
        "num_questions": len(qa_pairs),
        "questions": [],
        "strategy_metrics": {s.name: [] for s in strategies},
    }

    for q_idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        answer_sentence = qa.get("answer_sentence", qa.get("answer", ""))
        
        question_result = {
            "question": question,
            "answer": qa.get("answer", ""),
            "answer_sentence": answer_sentence,
            "strategies": {},
        }

        for strategy in strategies:
            nodes = strategy.retrieve(question)
            
            context = " ".join(n.node.text for n in nodes)
            retrieved_texts = [n.node.text for n in nodes]
            token_count = count_tokens(context)
            coherence = calculate_coherence(nodes)
            ir_metrics = compute_all_metrics(retrieved_texts, answer_sentence, k=5)

            question_result["strategies"][strategy.name] = {
                "token_count": token_count,
                "coherence_score": round(coherence, 3),
                "num_chunks": len(nodes),
                **{k: round(v, 3) for k, v in ir_metrics.items()},
            }
            
            article_result["strategy_metrics"][strategy.name].append({
                "token_count": token_count,
                "coherence": coherence,
                **ir_metrics,
            })

        article_result["questions"].append(question_result)

    return article_result


def run_benchmark():
    """Run benchmark comparing all strategies."""
    args = parse_args()

    print("=" * 60)
    print("Dynamic Semantic Window Benchmark")
    print("=" * 60)

    # Configure embedding model
    embed_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"\nLoading embedding model: {embed_model_name}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Setup results structure
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path(__file__).parent / "logs" / run_timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": embed_model_name,
            "similarity_threshold": os.getenv("SIMILARITY_THRESHOLD", "0.6"),
            "max_expand": os.getenv("MAX_EXPAND", "5"),
            "top_k": 5,
            "num_questions_per_article": args.num_questions,
            "num_articles": args.num_articles if args.source == "wikipedia" else 1,
        },
        "articles": [],
    }

    # Aggregate metrics across all articles
    all_metrics: dict[str, list[dict]] = {}

    if args.source == "wikipedia":
        from src.question_generator import generate_qa_pairs
        from src.wikipedia_loader import fetch_article_by_title, fetch_random_article

        num_articles = 1 if args.article else args.num_articles
        
        for article_idx in range(num_articles):
            print(f"\n{'=' * 60}")
            print(f"Article {article_idx + 1}/{num_articles}")
            print("=" * 60)

            try:
                if args.article:
                    print(f"Fetching: {args.article}")
                    title, text = fetch_article_by_title(args.article)
                else:
                    print("Fetching random Wikipedia article...")
                    title, text = fetch_random_article()

                print(f"📰 {title} ({len(text)} chars)")
                
                # Generate QA pairs
                print(f"Generating {args.num_questions} QA pairs...")
                qa_pairs = generate_qa_pairs(text, num_questions=args.num_questions)
                print(f"✅ Generated {len(qa_pairs)} QA pairs")

                if not qa_pairs:
                    print("⚠️ No QA pairs generated, skipping article")
                    continue

                # Initialize strategies for this article
                documents = [Document(text=text)]
                strategies = [
                    NaiveChunkingStrategy(documents, top_k=5),
                    FixedWindowStrategy(documents, top_k=5),
                    DynamicSemanticStrategy(
                        documents,
                        top_k=10,  # More seeds for dynamic expansion
                        threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.6")),
                        max_expand=int(os.getenv("MAX_EXPAND", "5")),
                    ),
                ]

                # Initialize aggregate metrics dict on first article
                if not all_metrics:
                    all_metrics = {s.name: [] for s in strategies}

                # Run benchmark
                print(f"\nBenchmarking {len(qa_pairs)} questions...")
                for q_idx, qa in enumerate(qa_pairs):
                    q_short = qa["question"][:50] + "..." if len(qa["question"]) > 50 else qa["question"]
                    metrics_str = []
                    
                    for strategy in strategies:
                        nodes = strategy.retrieve(qa["question"])
                        retrieved_texts = [n.node.text for n in nodes]
                        answer_sentence = qa.get("answer_sentence", qa.get("answer", ""))
                        
                        ir_metrics = compute_all_metrics(retrieved_texts, answer_sentence, k=5)
                        token_count = count_tokens(" ".join(retrieved_texts))
                        coherence = calculate_coherence(nodes)
                        
                        all_metrics[strategy.name].append({
                            "token_count": token_count,
                            "coherence": coherence,
                            **ir_metrics,
                        })
                        
                        metrics_str.append(f"{strategy.name[:3]}:HR={ir_metrics['hr@5']:.0f}")
                    
                    print(f"  Q{q_idx + 1}: {' | '.join(metrics_str)}")

                # Save article to results
                results["articles"].append({
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "num_questions": len(qa_pairs),
                })

            except Exception as e:
                print(f"❌ Error processing article: {e}")
                continue

    else:
        # Static mode
        data_path = Path(__file__).parent / "data" / "source_text.txt"
        print(f"Loading test data from: {data_path}")
        text = load_text_file(str(data_path))
        
        documents = [Document(text=text)]
        strategies = [
            NaiveChunkingStrategy(documents, top_k=5),
            FixedWindowStrategy(documents, top_k=5),
            DynamicSemanticStrategy(
                documents,
                top_k=10,  # More seeds for dynamic expansion
                threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.6")),
                max_expand=int(os.getenv("MAX_EXPAND", "5")),
            ),
        ]
        
        all_metrics = {s.name: [] for s in strategies}
        
        for qa in DEFAULT_QA_PAIRS:
            for strategy in strategies:
                nodes = strategy.retrieve(qa["question"])
                retrieved_texts = [n.node.text for n in nodes]
                answer_sentence = qa.get("answer_sentence", "")
                
                ir_metrics = compute_all_metrics(retrieved_texts, answer_sentence, k=5)
                token_count = count_tokens(" ".join(retrieved_texts))
                coherence = calculate_coherence(nodes)
                
                all_metrics[strategy.name].append({
                    "token_count": token_count,
                    "coherence": coherence,
                    **ir_metrics,
                })

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"benchmark_{run_timestamp}.json"

    # Add aggregate metrics to results
    results["aggregate_metrics"] = {}
    for strategy_name, metrics_list in all_metrics.items():
        if metrics_list:
            results["aggregate_metrics"][strategy_name] = {
                "avg_tokens": float(np.mean([m["token_count"] for m in metrics_list])),
                "avg_coherence": float(np.mean([m["coherence"] for m in metrics_list])),
                "avg_hr@5": float(np.mean([m["hr@5"] for m in metrics_list])),
                "avg_mrr": float(np.mean([m["mrr"] for m in metrics_list])),
                "avg_precision@5": float(np.mean([m["precision@5"] for m in metrics_list])),
                "avg_ndcg@5": float(np.mean([m["ndcg@5"] for m in metrics_list])),
                "num_questions": len(metrics_list),
            }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 90)
    print(f"BENCHMARK COMPLETE")
    print(f"Articles: {len(results['articles']) if results['articles'] else 1}")
    print(f"Total questions: {sum(len(m) for m in all_metrics.values()) // len(all_metrics) if all_metrics else 0}")
    print(f"Results saved to: {results_path}")
    print("=" * 90)

    print("\n📊 AGGREGATE RESULTS:")
    print("-" * 90)
    print(f"{'Strategy':20} | {'Tokens':>7} | {'Coher':>5} | {'HR@5':>5} | {'MRR':>5} | {'P@5':>5} | {'NDCG':>5}")
    print("-" * 90)

    for strategy_name, agg in results.get("aggregate_metrics", {}).items():
        print(f"{strategy_name:20} | {agg['avg_tokens']:7.1f} | {agg['avg_coherence']:5.3f} | "
              f"{agg['avg_hr@5']:5.2f} | {agg['avg_mrr']:5.2f} | {agg['avg_precision@5']:5.2f} | {agg['avg_ndcg@5']:5.2f}")


if __name__ == "__main__":
    run_benchmark()
