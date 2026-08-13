"""Evaluate specific parameters on the cached corpus.

Quick script to see what score any parameter set would get on the Optuna corpus.
"""

import pickle
from pathlib import Path

from src.config import DEFAULT_EXPANSION_CONFIG
from src.corpus_data import CorpusData
from src.expansion_core import (
    DynamicExpansionCore,
    build_garbage_mask,
    build_header_mask,
    evaluate_retrieval,
)

# Parameter sets to evaluate
PARAMS_TO_TEST = {
    "original": {
        "threshold": 0.85,
        "skip_threshold": 0.85,
        "min_window": 3,
        "max_expand": 7,
        "relevance_threshold_pct": 0.70,
        "merge_gap": 2,
    },
    "baseline": {
        "threshold": 0.85,
        "skip_threshold": 0.85,
        "min_window": 1,
        "max_expand": 7,
        "relevance_threshold_pct": 0.70,
        "merge_gap": 2,
    },
    "mrr_opt": {
        "threshold": 0.8342,
        "skip_threshold": 0.8684,
        "min_window": 1,
        "max_expand": 3,
        "relevance_threshold_pct": 0.7422,
        "merge_gap": 3,
    },
}


def compute_ndcg(rank: int, k: int = 5) -> float:
    """Compute NDCG@k for a single query with one relevant item."""
    if rank <= 0 or rank > k:
        return 0.0
    # DCG = 1 / log2(rank + 1), IDCG = 1 (best case rank=1)
    import math
    return 1.0 / math.log2(rank + 1)


def evaluate_params(corpus: CorpusData, params: dict) -> dict:
    """Evaluate a parameter set on the corpus."""
    article_lookup = {a.article_id: a for a in corpus.articles}
    garbage_masks = {
        a.article_id: build_garbage_mask(
            a.sentences, DEFAULT_EXPANSION_CONFIG.min_chunk_length
        )
        for a in corpus.articles
    }
    header_masks = {
        a.article_id: build_header_mask(a.sentences)
        for a in corpus.articles
    }

    total_hits = 0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_tokens = 0
    num_valid = 0
    
    for question in corpus.questions:
        article = article_lookup.get(question.article_id)
        if not article:
            continue
        
        expander = DynamicExpansionCore(
            neighbor_sims=article.neighbor_sims,
            sentence_sims=question.sentence_sims,
            top_k_indices=question.top_k_indices,
            threshold=params["threshold"],
            skip_threshold=params["skip_threshold"],
            min_window=params["min_window"],
            max_expand=params["max_expand"],
            relevance_threshold_pct=params["relevance_threshold_pct"],
            merge_gap=params["merge_gap"],
            target_clusters=5,
            garbage_mask=garbage_masks.get(question.article_id),
            header_mask=header_masks.get(question.article_id),
        )
        
        clusters = expander.expand_and_retrieve()
        metrics = evaluate_retrieval(
            clusters=clusters,
            answer_sentence_idx=question.answer_sentence_idx,
            sentences=article.sentences,
        )
        
        if question.answer_sentence_idx >= 0:
            total_hits += int(metrics["hit"])
            if metrics["hit"] and metrics["rank"] > 0:
                total_mrr += 1.0 / metrics["rank"]
                total_ndcg += compute_ndcg(metrics["rank"])
            num_valid += 1
        total_tokens += metrics["tokens"]
    
    if num_valid == 0:
        return {"hr": 0, "mrr": 0, "ndcg": 0, "tokens": 0}
    
    return {
        "hr": total_hits / num_valid,
        "mrr": total_mrr / num_valid,
        "ndcg": total_ndcg / num_valid,
        "tokens": total_tokens / len(corpus.questions),
    }


def main():
    # Load corpus
    corpus_path = Path("data/cached_corpus.pkl")
    if not corpus_path.exists():
        print("❌ Corpus not found. Run prepare_corpus.py first.")
        return
    
    print("📦 Loading corpus...")
    with open(corpus_path, "rb") as f:
        corpus: CorpusData = pickle.load(f)
    
    valid = sum(1 for q in corpus.questions if q.answer_sentence_idx >= 0)
    print(f"   ✅ {len(corpus.articles)} articles, {valid} valid questions")
    
    print("\n" + "=" * 80)
    print("📊 PARAMETER EVALUATION ON CACHED CORPUS")
    print("=" * 80)
    print(f"{'Config':<15} | {'HR':>8} | {'MRR':>8} | {'NDCG':>8} | {'Tokens':>8}")
    print("-" * 80)
    
    for name, params in PARAMS_TO_TEST.items():
        result = evaluate_params(corpus, params)
        print(f"{name:<15} | {result['hr']:>8.4f} | {result['mrr']:>8.4f} | {result['ndcg']:>8.4f} | {result['tokens']:>8.1f}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
