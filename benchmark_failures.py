"""Re-benchmark on failures data - ONLY failed questions."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.metrics import compute_all_metrics
from src.strategies import (
    DynamicSemanticStrategy,
    FixedWindowStrategy,
    NaiveChunkingStrategy,
    SemanticSplitterStrategy,
)

load_dotenv()

# Load embedding model
model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
print(f"Loading: {model_name}")
Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)


def count_tokens(text: str) -> int:
    """Rough token count (chars / 4)."""
    return len(text) // 4


def load_failures() -> list[dict]:
    """Load all failure files - extract ONLY the failed question."""
    failures_dir = Path(__file__).parent / "results" / "failures"
    
    failures = []
    
    for failure_file in failures_dir.glob("*.json"):
        try:
            with open(failure_file, encoding="utf-8") as f:
                data = json.load(f)
            
            full_text = data.get("full_text")
            question = data.get("question")
            answer = data.get("expected_answer")
            title = data.get("article_title", "Unknown")
            
            if not full_text or not question or not answer:
                continue
            
            failures.append({
                "title": title,
                "text": full_text,
                "question": question,
                "answer": answer,
            })
        except Exception as e:
            print(f"Error loading {failure_file.name}: {e}")
    
    return failures


def main():
    """Run benchmark on failed questions only."""
    print("=" * 60)
    print("Re-benchmark on Failed Questions ONLY")
    print("=" * 60)
    
    # Load failures
    print("\n📥 Loading failures data...")
    failures = load_failures()
    print(f"   Loaded {len(failures)} failed questions")
    
    # Group by article text (to avoid rebuilding index)
    articles = {}
    for f in failures:
        key = hash(f["text"][:1000])  # Use text prefix as key
        if key not in articles:
            articles[key] = {
                "title": f["title"],
                "text": f["text"],
                "questions": [],
            }
        articles[key]["questions"].append({
            "question": f["question"],
            "answer": f["answer"],
        })
    
    print(f"   Unique articles: {len(articles)}")
    
    # Run benchmark
    print(f"\n🔬 Benchmarking {len(failures)} failed questions...")
    
    strategy_metrics = {
        "Naive Chunking": {"hr": [], "mrr": [], "tokens": [], "num_chunks": []},
        "Fixed Window": {"hr": [], "mrr": [], "tokens": [], "num_chunks": []},
        "Semantic Splitter": {"hr": [], "mrr": [], "tokens": [], "num_chunks": []},
        "Dynamic Semantic": {"hr": [], "mrr": [], "tokens": [], "num_chunks": []},
    }
    
    for i, (_key, article) in enumerate(articles.items()):
        print(f"  [{i+1}/{len(articles)}] {article['title'][:50]}... ({len(article['questions'])} q)")
        
        documents = [Document(text=article["text"])]
        
        strategies = [
            NaiveChunkingStrategy(documents, top_k=5),
            FixedWindowStrategy(documents, top_k=5),
            SemanticSplitterStrategy(documents, top_k=5),
            DynamicSemanticStrategy(documents, top_k=5),
        ]
        
        for qa in article["questions"]:
            question = qa["question"]
            answer = qa["answer"]
            
            for strategy in strategies:
                nodes = strategy.retrieve(question)
                texts = [n.node.text for n in nodes]
                
                metrics = compute_all_metrics(texts, answer, k=5)
                metrics["tokens"] = count_tokens(" ".join(texts))
                metrics["num_chunks"] = len(nodes)
                
                strategy_metrics[strategy.name]["hr"].append(metrics["hr@5"])
                strategy_metrics[strategy.name]["mrr"].append(metrics["mrr"])
                strategy_metrics[strategy.name]["tokens"].append(metrics["tokens"])
                strategy_metrics[strategy.name]["num_chunks"].append(metrics["num_chunks"])
    
    # Print results
    print(f"\n📊 RESULTS ({len(failures)} FAILED questions)")
    print("-" * 75)
    print(f"{'Strategy':<20} | {'Tokens':>7} | {'HR@5':>6} | {'MRR':>6} | {'Chunks':>6}")
    print("-" * 75)
    
    for name in ["Naive Chunking", "Fixed Window", "Semantic Splitter", "Dynamic Semantic"]:
        m = strategy_metrics[name]
        if m["hr"]:
            avg_tokens = sum(m["tokens"]) / len(m["tokens"])
            avg_hr = sum(m["hr"]) / len(m["hr"])
            avg_mrr = sum(m["mrr"]) / len(m["mrr"])
            avg_chunks = sum(m["num_chunks"]) / len(m["num_chunks"])
            print(f"{name:<20} | {avg_tokens:>7.1f} | {avg_hr:>6.2f} | {avg_mrr:>6.2f} | {avg_chunks:>6.2f}")
    
    print("-" * 75)
    
    # Count how many Dynamic Semantic now hits
    ds_hits = sum(strategy_metrics["Dynamic Semantic"]["hr"])
    print(f"\n✅ Dynamic Semantic now hits: {int(ds_hits)}/{len(failures)} ({ds_hits/len(failures)*100:.1f}%)")


if __name__ == "__main__":
    main()
