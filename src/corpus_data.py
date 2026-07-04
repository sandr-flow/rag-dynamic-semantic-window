"""Data structures for pre-computed corpus used in Optuna optimization."""

import warnings
from dataclasses import dataclass, field

import numpy as np

from src.answer_matching import find_answer_sentence_idx as _find_answer_sentence_idx


@dataclass
class ArticleData:
    """Pre-computed data for a single article.
    
    Attributes:
        article_id: Unique identifier for the article.
        title: Article title.
        sentences: List of raw sentence strings.
        embeddings: Sentence embeddings, shape (num_sentences, embed_dim).
            Built in the corpus embedding mode (phantom texts when
            phantom_window > 0); serve the question sentence_sims.
        neighbor_sims: Cosine similarity between adjacent sentences,
                       shape (num_sentences-1,). neighbor_sims[i] = cos(S_i, S_{i+1}).
                       Computed in ``CorpusData.adjacency_space``.
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
    source: str = "unknown"
    embedding_provider: str = "unknown"
    embedding_model: str = "unknown"
    phantom_window: int = 0
    embedding_mode: str = "sentence"
    adjacency_space: str = "phantom"


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
    return _find_answer_sentence_idx(sentences, answer)


def subset_corpus_by_article_ids(corpus: CorpusData, article_ids: set[int]) -> CorpusData:
    """Return a corpus view containing only selected articles and their questions."""
    selected = set(article_ids)
    return CorpusData(
        articles=[article for article in corpus.articles if article.article_id in selected],
        questions=[
            question for question in corpus.questions if question.article_id in selected
        ],
        embed_dim=corpus.embed_dim,
        top_k=corpus.top_k,
        source=corpus.source,
        embedding_provider=corpus.embedding_provider,
        embedding_model=corpus.embedding_model,
        phantom_window=corpus.phantom_window,
        embedding_mode=corpus.embedding_mode,
        adjacency_space=corpus.adjacency_space,
    )


def split_corpus_by_documents(
    corpus: CorpusData,
    *,
    train_ratio: float = 0.70,
    seed: int = 42,
) -> tuple[CorpusData, CorpusData]:
    """Split a corpus into train/validation subsets by article id."""
    article_ids = [article.article_id for article in corpus.articles]
    if len(article_ids) < 2:
        warnings.warn(
            "corpus has fewer than 2 articles: train/validation split is "
            "disabled, validation metrics will equal train metrics (full "
            "leakage)",
            stacklevel=2,
        )
        return corpus, corpus

    rng = np.random.default_rng(seed)
    shuffled = list(article_ids)
    rng.shuffle(shuffled)
    train_size = round(len(shuffled) * train_ratio)
    train_size = max(1, min(len(shuffled) - 1, train_size))

    train_ids = set(shuffled[:train_size])
    val_ids = set(shuffled[train_size:])
    return (
        subset_corpus_by_article_ids(corpus, train_ids),
        subset_corpus_by_article_ids(corpus, val_ids),
    )
