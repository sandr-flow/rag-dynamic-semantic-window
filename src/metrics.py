"""Retrieval quality metrics for benchmark evaluation."""

import math

from src.answer_matching import contains_answer


def _contains_answer(chunk_text: str, answer_sentence: str) -> bool:
    """
    Check if chunk contains the answer sentence.

    Uses substring matching with normalization.
    """
    return contains_answer(chunk_text, answer_sentence)


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
    retrieved_texts: list[str], answer_sentence: str, k: int | None = None
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
    retrieved_texts: list[str], answer_sentence: str, k: int | None = None
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
    retrieved_texts: list[str], answer_sentence: str, k: int | None = None
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


def joint_hit_rate(retrieved_texts: list[str], answer_sentences: list[str], k: int) -> float:
    """1.0 when every answer sentence appears in top-k chunks."""
    if not answer_sentences:
        return 0.0
    top_k = retrieved_texts[:k]
    return float(
        all(any(_contains_answer(chunk, answer) for chunk in top_k) for answer in answer_sentences)
    )


def partial_hit_rate(retrieved_texts: list[str], answer_sentences: list[str], k: int) -> float:
    """1.0 when at least one answer sentence appears in top-k chunks."""
    if not answer_sentences:
        return 0.0
    top_k = retrieved_texts[:k]
    return float(
        any(
            any(_contains_answer(chunk, answer) for chunk in top_k)
            for answer in answer_sentences
        )
    )


def answer_recall_at_k(retrieved_texts: list[str], answer_sentences: list[str], k: int) -> float:
    """Fraction of answer sentences found in top-k chunks."""
    if not answer_sentences:
        return 0.0
    top_k = retrieved_texts[:k]
    hits = sum(
        1
        for answer in answer_sentences
        if any(_contains_answer(chunk, answer) for chunk in top_k)
    )
    return hits / len(answer_sentences)


def mrr_multi(retrieved_texts: list[str], answer_sentences: list[str]) -> float:
    """Mean per-answer MRR."""
    if not answer_sentences:
        return 0.0
    return sum(mrr(retrieved_texts, answer) for answer in answer_sentences) / len(answer_sentences)


def mrr_min_multi(retrieved_texts: list[str], answer_sentences: list[str]) -> float:
    """Minimum per-answer MRR (weakest answer dominates)."""
    if not answer_sentences:
        return 0.0
    return min(mrr(retrieved_texts, answer) for answer in answer_sentences)


def precision_at_k_multi(
    retrieved_texts: list[str], answer_sentences: list[str], k: int
) -> float:
    """Fraction of top-k slots that contain any still-uncovered answer."""
    top_k = retrieved_texts[:k]
    if not top_k:
        return 0.0
    hits = 0
    found: set[int] = set()
    for chunk in top_k:
        for idx, answer in enumerate(answer_sentences):
            if idx in found:
                continue
            if _contains_answer(chunk, answer):
                hits += 1
                found.add(idx)
                break
    return hits / k


def ndcg_at_k_multi(
    retrieved_texts: list[str], answer_sentences: list[str], k: int
) -> float:
    """NDCG@k with one relevant hit per distinct answer sentence."""
    top_k = retrieved_texts[:k]
    if not top_k or not answer_sentences:
        return 0.0

    relevances = []
    found: set[int] = set()
    for chunk in top_k:
        rel = 0.0
        for idx, answer in enumerate(answer_sentences):
            if idx in found:
                continue
            if _contains_answer(chunk, answer):
                rel = 1.0
                found.add(idx)
                break
        relevances.append(rel)

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    num_relevant = min(len(answer_sentences), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def compute_multi_answer_metrics(
    retrieved_texts: list[str], answer_sentences: list[str], k: int = 5
) -> dict[str, float]:
    """Compute retrieval metrics for compound questions with multiple answers."""
    return {
        f"hr@{k}": joint_hit_rate(retrieved_texts, answer_sentences, k),
        f"partial_hr@{k}": partial_hit_rate(retrieved_texts, answer_sentences, k),
        f"answer_recall@{k}": answer_recall_at_k(retrieved_texts, answer_sentences, k),
        "mrr": mrr_multi(retrieved_texts, answer_sentences),
        "mrr_min": mrr_min_multi(retrieved_texts, answer_sentences),
        f"precision@{k}": precision_at_k_multi(retrieved_texts, answer_sentences, k),
        f"ndcg@{k}": ndcg_at_k_multi(retrieved_texts, answer_sentences, k),
    }


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


def recall_at_k_ids(
    retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None
) -> float:
    """
    Calculate Recall@K for ID-based retrieval.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked).
        relevant_ids: List of relevant document IDs (ground truth).
        k: Cutoff rank.
        
    Returns:
        Recall score (fraction of relevant docs found).
    """
    if not relevant_ids:
        return 0.0
        
    if k is None:
        k = len(retrieved_ids)
        
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = top_k.intersection(relevant_set)
    return len(hits) / len(relevant_set)


def ndcg_at_k_ids(
    retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None
) -> float:
    """
    Calculate NDCG@K for ID-based retrieval (binary relevance).
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked).
        relevant_ids: List of relevant document IDs.
        k: Cutoff rank.
        
    Returns:
        NDCG score.
    """
    if not relevant_ids:
        return 0.0
        
    if k is None:
        k = len(retrieved_ids)
        
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(len(relevant_set), k)):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0
