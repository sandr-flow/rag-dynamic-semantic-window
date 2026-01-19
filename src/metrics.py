"""Retrieval quality metrics for benchmark evaluation."""

import math
from typing import Optional


def _contains_answer(chunk_text: str, answer_sentence: str) -> bool:
    """
    Check if chunk contains the answer sentence.

    Uses substring matching with normalization.
    """
    # Normalize whitespace for comparison
    chunk_norm = " ".join(chunk_text.lower().split())
    answer_norm = " ".join(answer_sentence.lower().split())
    return answer_norm in chunk_norm


def hit_rate(retrieved_texts: list[str], answer_sentence: str) -> float:
    """
    Calculate Hit Rate (HR@K).

    Returns 1.0 if answer_sentence is found in any retrieved chunk, else 0.0.

    Args:
        retrieved_texts: List of retrieved chunk texts.
        answer_sentence: The sentence containing the expected answer.

    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    for chunk in retrieved_texts:
        if _contains_answer(chunk, answer_sentence):
            return 1.0
    return 0.0


def mrr(retrieved_texts: list[str], answer_sentence: str) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Returns 1/rank of the first chunk containing the answer, or 0.0 if not found.

    Args:
        retrieved_texts: List of retrieved chunk texts (ordered by rank).
        answer_sentence: The sentence containing the expected answer.

    Returns:
        1/rank or 0.0.
    """
    for i, chunk in enumerate(retrieved_texts):
        if _contains_answer(chunk, answer_sentence):
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    retrieved_texts: list[str], answer_sentence: str, k: Optional[int] = None
) -> float:
    """
    Calculate Precision@K.

    Fraction of top-k chunks that contain the answer.

    Args:
        retrieved_texts: List of retrieved chunk texts.
        answer_sentence: The sentence containing the expected answer.
        k: Number of top results to consider (default: all).

    Returns:
        Precision score.
    """
    if k is None:
        k = len(retrieved_texts)
    
    top_k = retrieved_texts[:k]
    if not top_k:
        return 0.0
    
    hits = sum(1 for chunk in top_k if _contains_answer(chunk, answer_sentence))
    return hits / k


def recall_at_k(
    retrieved_texts: list[str], answer_sentence: str, k: Optional[int] = None
) -> float:
    """
    Calculate Recall@K.

    For single relevant document (answer_sentence), this equals Hit Rate.

    Args:
        retrieved_texts: List of retrieved chunk texts.
        answer_sentence: The sentence containing the expected answer.
        k: Number of top results to consider (default: all).

    Returns:
        1.0 if answer in top-k, else 0.0.
    """
    if k is None:
        k = len(retrieved_texts)
    
    top_k = retrieved_texts[:k]
    return hit_rate(top_k, answer_sentence)


def ndcg_at_k(
    retrieved_texts: list[str], answer_sentence: str, k: Optional[int] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Uses binary relevance: 1 if chunk contains answer, 0 otherwise.

    Args:
        retrieved_texts: List of retrieved chunk texts.
        answer_sentence: The sentence containing the expected answer.
        k: Number of top results to consider (default: all).

    Returns:
        NDCG score in range [0, 1].
    """
    if k is None:
        k = len(retrieved_texts)
    
    top_k = retrieved_texts[:k]
    if not top_k:
        return 0.0
    
    # Calculate relevance for each chunk
    relevances = [1.0 if _contains_answer(chunk, answer_sentence) else 0.0 for chunk in top_k]
    
    # Calculate DCG
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Ideal DCG: all relevant docs at top positions
    num_relevant = sum(relevances)
    if num_relevant == 0:
        return 0.0
    
    idcg = 0.0
    for i in range(int(num_relevant)):
        idcg += 1.0 / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(
    retrieved_texts: list[str], answer_sentence: str, k: int = 5
) -> dict[str, float]:
    """
    Compute all retrieval metrics.

    Args:
        retrieved_texts: List of retrieved chunk texts.
        answer_sentence: The sentence containing the expected answer.
        k: K value for @K metrics.

    Returns:
        Dict with all metric scores.
    """
    return {
        f"hr@{k}": hit_rate(retrieved_texts[:k], answer_sentence),
        "mrr": mrr(retrieved_texts, answer_sentence),
        f"precision@{k}": precision_at_k(retrieved_texts, answer_sentence, k),
        f"recall@{k}": recall_at_k(retrieved_texts, answer_sentence, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_texts, answer_sentence, k),
    }
