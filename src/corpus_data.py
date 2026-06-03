"""Data structures for pre-computed corpus used in Optuna optimization."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArticleData:
    """Pre-computed data for a single article.
    
    Attributes:
        article_id: Unique identifier for the article.
        title: Article title.
        sentences: List of raw sentence strings.
        embeddings: Sentence embeddings, shape (num_sentences, embed_dim).
        neighbor_sims: Cosine similarity between adjacent sentences,
                       shape (num_sentences-1,). neighbor_sims[i] = cos(S_i, S_{i+1}).
    """
    
    article_id: int
    title: str
    sentences: list[str]
    embeddings: np.ndarray  # Shape: (num_sentences, embed_dim)
    neighbor_sims: np.ndarray  # Shape: (num_sentences-1,)


@dataclass
class QuestionData:
    """Pre-computed data for a single question.
    
    Attributes:
        question_id: Unique identifier for the question.
        article_id: ID of the source article.
        question: Question text.
        answer_sentence: Expected answer sentence from the article.
        answer_sentence_idx: Index of answer sentence in article.sentences (or -1 if not found).
        embedding: Question embedding, shape (embed_dim,).
        sentence_sims: Cosine similarity to all sentences in the article,
                       shape (num_sentences,). sentence_sims[i] = cos(Q, S_i).
        top_k_indices: Top-K sentence indices by similarity (pre-computed retrieval).
    """
    
    question_id: int
    article_id: int
    question: str
    answer_sentence: str
    answer_sentence_idx: int
    embedding: np.ndarray  # Shape: (embed_dim,)
    sentence_sims: np.ndarray  # Shape: (num_sentences,)
    top_k_indices: np.ndarray  # Shape: (top_k,)


@dataclass
class CorpusData:
    """Complete pre-computed corpus for Optuna optimization.
    
    All embeddings and similarity matrices are pre-computed.
    During Optuna trials, only numpy array lookups are performed.
    
    Attributes:
        articles: List of article data.
        questions: List of question data.
        embed_dim: Embedding dimension.
        top_k: Number of pre-computed top candidates per question.
    """
    
    articles: list[ArticleData] = field(default_factory=list)
    questions: list[QuestionData] = field(default_factory=list)
    embed_dim: int = 384  # Default for bge-small-en-v1.5
    top_k: int = 100


def find_answer_sentence_idx(sentences: list[str], answer: str) -> int:
    """
    Find index of the sentence containing the answer.
    
    Uses substring matching with normalization.
    
    Args:
        sentences: List of sentences.
        answer: Answer string to find.
    
    Returns:
        Index of the matching sentence, or -1 if not found.
    """
    answer_normalized = answer.lower().strip()
    
    # First try exact substring match
    for i, sent in enumerate(sentences):
        if answer_normalized in sent.lower():
            return i
    
    # Try token overlap for fuzzy matching
    answer_tokens = set(answer_normalized.split())
    best_idx = -1
    best_overlap = 0.0
    
    for i, sent in enumerate(sentences):
        sent_tokens = set(sent.lower().split())
        if not answer_tokens:
            continue
        overlap = len(answer_tokens & sent_tokens) / len(answer_tokens)
        if overlap > best_overlap and overlap >= 0.7:
            best_overlap = overlap
            best_idx = i
    
    return best_idx
