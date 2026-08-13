"""Data structures for pre-computed corpus used in Optuna optimization."""

import warnings
from dataclasses import dataclass, field

import numpy as np

from src.answer_matching import find_answer_sentence_idx as _find_answer_sentence_idx
from src.seed_retrieval import dual_seed_indices


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
        clean_embeddings: Per-sentence embeddings used for dual-seed ranking
            (Fixed Window matching). None when phantom_window=0.
    """

    article_id: int
    title: str
    sentences: list[str]
    embeddings: np.ndarray  # Shape: (num_sentences, embed_dim)
    neighbor_sims: np.ndarray  # Shape: (num_sentences-1,)
    clean_embeddings: np.ndarray | None = None


@dataclass
class QuestionData:
    """Pre-computed data for a single question.
    
    Attributes:
        question_id: Unique identifier for the question.
        article_id: ID of the source article (per-document) or first source doc
            (shared / extrahard compound questions).
        question: Question text.
        answer_sentence: Expected answer sentence from the article.
        answer_sentence_idx: Index of answer sentence in article.sentences (or -1 if not found).
        embedding: Question embedding, shape (embed_dim,).
        sentence_sims: Cosine similarity to all sentences in the article,
                       shape (num_sentences,). sentence_sims[i] = cos(Q, S_i).
        top_k_indices: Candidate seed indices in rank-interleave order
            (phantom ∪ clean when dual-seed is on; length ≤ 2 * top_k).
        source_article_ids: For shared-index compound questions, article ids of
            every answer-bearing document (empty for per-document questions).
        answer_local_indices: Parallel local sentence indices inside each
            ``source_article_ids`` article (-1 when not found).
    """
    
    question_id: int
    article_id: int
    question: str
    answer_sentence: str
    answer_sentence_idx: int
    embedding: np.ndarray  # Shape: (embed_dim,)
    sentence_sims: np.ndarray  # Shape: (num_sentences,)
    top_k_indices: np.ndarray  # Shape: (top_k,)
    source_article_ids: list[int] = field(default_factory=list)
    answer_local_indices: list[int] = field(default_factory=list)


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
    # Shared-index fields (extrahard cross-document tuning / evaluation).
    kind: str = "per_document"
    global_sentences: list[str] = field(default_factory=list)
    global_neighbor_sims: np.ndarray | None = None
    global_garbage_mask: np.ndarray | None = None
    global_segment_ids: np.ndarray | None = None
    article_offsets: dict[int, int] = field(default_factory=dict)


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


def is_shared_corpus(corpus: CorpusData) -> bool:
    return corpus.kind == "shared"


def global_answer_indices(question: QuestionData, corpus: CorpusData) -> list[int]:
    """Map per-article answer locations to global sentence indices."""
    if not question.source_article_ids:
        if question.answer_sentence_idx < 0:
            return []
        offset = corpus.article_offsets.get(question.article_id, 0)
        return [offset + question.answer_sentence_idx]

    indices: list[int] = []
    for article_id, local_idx in zip(
        question.source_article_ids, question.answer_local_indices, strict=True
    ):
        if local_idx < 0:
            indices.append(-1)
            continue
        offset = corpus.article_offsets.get(article_id)
        if offset is None:
            indices.append(-1)
            continue
        indices.append(offset + local_idx)
    return indices


def question_is_valid(question: QuestionData, corpus: CorpusData) -> bool:
    """True when every expected answer sentence was located in the corpus."""
    if is_shared_corpus(corpus) and question.source_article_ids:
        indices = global_answer_indices(question, corpus)
        return bool(indices) and all(idx >= 0 for idx in indices)
    return question.answer_sentence_idx >= 0


def build_shared_global_arrays(
    articles: list[ArticleData],
    items: list[dict],
    *,
    min_chunk_length: int,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, int],
]:
    """Concatenate per-article arrays into one shared index."""
    from src.expansion_core import build_garbage_mask

    global_sentences: list[str] = []
    neighbor_parts: list[np.ndarray] = []
    garbage_parts: list[np.ndarray] = []
    segment_parts: list[np.ndarray] = []
    article_offsets: dict[int, int] = {}

    for pos, article in enumerate(articles):
        item = items[pos] if pos < len(items) else {"id": article.article_id}
        doc_id = str(item.get("id", article.article_id))
        article_offsets[article.article_id] = len(global_sentences)
        global_sentences.extend(article.sentences)
        if len(article.neighbor_sims):
            neighbor_parts.append(article.neighbor_sims)
        garbage_parts.append(
            build_garbage_mask(article.sentences, min_chunk_length)
        )
        segment_parts.append(
            np.full(len(article.sentences), doc_id, dtype=object)
        )
        if pos < len(articles) - 1:
            neighbor_parts.append(np.array([0.0], dtype=np.float32))

    global_neighbor_sims = (
        np.concatenate(neighbor_parts) if neighbor_parts else np.array([], dtype=np.float32)
    )
    global_garbage_mask = (
        np.concatenate(garbage_parts) if garbage_parts else np.array([], dtype=bool)
    )
    global_segment_ids = (
        np.concatenate(segment_parts) if segment_parts else np.array([], dtype=object)
    )
    return (
        global_sentences,
        global_neighbor_sims,
        global_garbage_mask,
        global_segment_ids,
        article_offsets,
    )


def _article_clean_embeddings(article: ArticleData) -> np.ndarray:
    if article.clean_embeddings is not None:
        return article.clean_embeddings
    return article.embeddings


def _concat_clean_embeddings(articles: list[ArticleData], embed_dim: int) -> np.ndarray:
    if not articles:
        return np.zeros((0, embed_dim), dtype=np.float32)
    return np.concatenate([_article_clean_embeddings(article) for article in articles], axis=0)


def _reindex_question_for_shared(
    question: QuestionData,
    corpus: CorpusData,
    global_embeddings: np.ndarray,
    global_clean_embeddings: np.ndarray | None = None,
) -> QuestionData:
    """Recompute query similarities after a shared corpus subset."""
    norms = np.linalg.norm(global_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = global_embeddings / norms
    q_norm = np.linalg.norm(question.embedding)
    q = question.embedding / q_norm if q_norm > 0 else question.embedding
    sentence_sims = normalized @ q
    k = min(corpus.top_k, len(sentence_sims))
    clean_sims = None
    if global_clean_embeddings is not None and len(global_clean_embeddings) == len(
        global_embeddings
    ):
        clean_norms = np.linalg.norm(global_clean_embeddings, axis=1, keepdims=True)
        clean_norms = np.where(clean_norms == 0, 1, clean_norms)
        clean_sims = (global_clean_embeddings / clean_norms) @ q
    top_k_indices = dual_seed_indices(sentence_sims, clean_sims, k)
    return QuestionData(
        question_id=question.question_id,
        article_id=question.article_id,
        question=question.question,
        answer_sentence=question.answer_sentence,
        answer_sentence_idx=question.answer_sentence_idx,
        embedding=question.embedding,
        sentence_sims=sentence_sims,
        top_k_indices=top_k_indices,
        source_article_ids=list(question.source_article_ids),
        answer_local_indices=list(question.answer_local_indices),
    )


def subset_corpus_by_article_ids(corpus: CorpusData, article_ids: set[int]) -> CorpusData:
    """Return a corpus view containing only selected articles and their questions."""
    from src.config import DEFAULT_EXPANSION_CONFIG

    selected = set(article_ids)
    articles = [article for article in corpus.articles if article.article_id in selected]
    if not is_shared_corpus(corpus):
        return CorpusData(
            articles=articles,
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
            kind=corpus.kind,
        )

    items_stub = [{"id": article.article_id} for article in articles]
    (
        global_sentences,
        global_neighbor_sims,
        global_garbage_mask,
        global_segment_ids,
        article_offsets,
    ) = build_shared_global_arrays(
        articles,
        items_stub,
        min_chunk_length=DEFAULT_EXPANSION_CONFIG.min_chunk_length,
    )

    global_embeddings = (
        np.concatenate([article.embeddings for article in articles], axis=0)
        if articles
        else np.zeros((0, corpus.embed_dim), dtype=np.float32)
    )

    questions: list[QuestionData] = []
    global_clean_embeddings = _concat_clean_embeddings(articles, corpus.embed_dim)
    for question in corpus.questions:
        # Compound questions carry every source doc; single-answer ones only
        # live in their own article. Both must survive a shared-corpus subset.
        source_ids = question.source_article_ids or [question.article_id]
        if not all(article_id in selected for article_id in source_ids):
            continue
        questions.append(
            _reindex_question_for_shared(
                question,
                corpus,
                global_embeddings,
                global_clean_embeddings=global_clean_embeddings,
            )
        )

    return CorpusData(
        articles=articles,
        questions=questions,
        embed_dim=corpus.embed_dim,
        top_k=corpus.top_k,
        source=corpus.source,
        embedding_provider=corpus.embedding_provider,
        embedding_model=corpus.embedding_model,
        phantom_window=corpus.phantom_window,
        embedding_mode=corpus.embedding_mode,
        adjacency_space=corpus.adjacency_space,
        kind="shared",
        global_sentences=global_sentences,
        global_neighbor_sims=global_neighbor_sims,
        global_garbage_mask=global_garbage_mask,
        global_segment_ids=global_segment_ids,
        article_offsets=article_offsets,
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
