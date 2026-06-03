"""Prepare corpus for Optuna hyperparameter optimization.

This script fetches Wikipedia articles, generates QA pairs, computes all embeddings
and similarity matrices once, then saves everything to disk for fast Optuna trials.

Usage:
    python prepare_corpus.py [--num-articles 100] [--min-length 10000]
"""

import argparse
import asyncio
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import DEFAULT_CORPUS_CONFIG
from src.corpus_data import ArticleData, CorpusData, QuestionData, find_answer_sentence_idx
from src.question_generator import generate_qa_pairs_async
from src.utils import split_into_sentences
from src.wikipedia_loader import fetch_random_articles_batch

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare corpus for Optuna optimization")
    parser.add_argument(
        "--source",
        choices=["wikipedia", "qasper"],
        default="wikipedia",
        help="Data source: wikipedia or qasper",
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=DEFAULT_CORPUS_CONFIG.num_articles,
        help="Number of articles to fetch",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_CORPUS_CONFIG.min_article_length,
        help="Minimum article length in characters",
    )
    parser.add_argument(
        "--questions-per-article",
        type=int,
        default=DEFAULT_CORPUS_CONFIG.questions_per_article,
        help="Number of QA pairs per article",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CORPUS_CONFIG.top_k_candidates,
        help="Number of top candidates to pre-compute",
    )
    return parser.parse_args()


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_neighbor_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between adjacent sentences.
    
    Args:
        embeddings: Shape (num_sentences, embed_dim)
    
    Returns:
        Shape (num_sentences-1,) where result[i] = cos(emb[i], emb[i+1])
    """
    if len(embeddings) < 2:
        return np.array([])
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms
    
    # Compute dot product of adjacent pairs
    neighbor_sims = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return neighbor_sims


def compute_question_sentence_sims(
    question_embedding: np.ndarray,
    sentence_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between question and all sentences.
    
    Args:
        question_embedding: Shape (embed_dim,)
        sentence_embeddings: Shape (num_sentences, embed_dim)
    
    Returns:
        Shape (num_sentences,) where result[i] = cos(q, s_i)
    """
    # Normalize question
    q_norm = np.linalg.norm(question_embedding)
    if q_norm == 0:
        return np.zeros(len(sentence_embeddings))
    q_normalized = question_embedding / q_norm
    
    # Normalize sentences
    s_norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    s_norms = np.where(s_norms == 0, 1, s_norms)
    s_normalized = sentence_embeddings / s_norms
    
    # Dot product
    sims = s_normalized @ q_normalized
    return sims


async def main():
    """Main corpus preparation pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("Corpus Preparation for Optuna Optimization")
    print("=" * 60)
    
    # Setup paths
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    articles_path = data_dir / "articles.jsonl"
    questions_path = data_dir / "questions.jsonl"
    corpus_path = data_dir / "cached_corpus.pkl"
    
    # Load embedding model
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"\n📦 Loading embedding model: {model_name}")
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.embed_model = embed_model
    
    # Get embedding dimension
    test_emb = embed_model.get_text_embedding("test")
    embed_dim = len(test_emb)
    print(f"   Embedding dimension: {embed_dim}")
    
    start_time = time.time()
    
    # Step 1: Fetch articles from selected source
    if args.source == "qasper":
        from src.qasper_loader import fetch_qasper_articles
        print(f"\n📥 Loading {args.num_articles} QASPER articles (min {args.min_length} chars)...")
        articles_raw = fetch_qasper_articles(args.num_articles, args.min_length)
    else:
        print(f"\n📥 Fetching {args.num_articles} Wikipedia articles (min {args.min_length} chars)...")
        articles_raw = await fetch_random_articles_batch(
            count=args.num_articles,
            min_length=args.min_length,
        )
    print(f"   ✅ Got {len(articles_raw)} articles")
    
    # Step 2: Save raw articles to JSONL
    print(f"\n💾 Saving raw articles to {articles_path}...")
    with open(articles_path, "w", encoding="utf-8") as f:
        for i, (title, text) in enumerate(articles_raw):
            json.dump({"id": i, "title": title, "text": text}, f, ensure_ascii=False)
            f.write("\n")
    
    # Step 3: Process articles - split sentences, compute embeddings
    print("\n🔬 Processing articles...")
    articles_data: list[ArticleData] = []
    
    for article_id, (title, text) in enumerate(articles_raw):
        print(f"   [{article_id + 1}/{len(articles_raw)}] {title[:50]}...")
        
        # Split into sentences
        sentences = split_into_sentences(text)
        if len(sentences) < 10:
            print(f"      ⚠️ Skipping (too few sentences: {len(sentences)})")
            continue
        
        # Compute sentence embeddings (batch)
        embeddings = np.array([
            embed_model.get_text_embedding(sent) for sent in sentences
        ], dtype=np.float32)
        
        # Compute neighbor similarities
        neighbor_sims = compute_neighbor_similarities(embeddings)
        
        articles_data.append(ArticleData(
            article_id=article_id,
            title=title,
            sentences=sentences,
            embeddings=embeddings,
            neighbor_sims=neighbor_sims,
        ))
    
    print(f"   ✅ Processed {len(articles_data)} articles")
    
    # Step 4: Generate QA pairs
    print(f"\n📝 Generating QA pairs ({args.questions_per_article} per article)...")
    all_qa_pairs = []
    
    for article in articles_data:
        # Join sentences back to text for QA generation
        text = " ".join(article.sentences)
        qa_pairs = await generate_qa_pairs_async(text, num_questions=args.questions_per_article)
        
        if qa_pairs:
            for qa in qa_pairs:
                all_qa_pairs.append({
                    "article_id": article.article_id,
                    "article_title": article.title,
                    "question": qa["question"],
                    "answer_sentence": qa.get("answer_sentence", qa.get("answer", "")),
                })
        
        # Rate limit for Mistral API
        await asyncio.sleep(1.1)
    
    print(f"   ✅ Generated {len(all_qa_pairs)} QA pairs")
    
    # Step 5: Save QA pairs to JSONL
    print(f"\n💾 Saving QA pairs to {questions_path}...")
    with open(questions_path, "w", encoding="utf-8") as f:
        for i, qa in enumerate(all_qa_pairs):
            qa["question_id"] = i
            json.dump(qa, f, ensure_ascii=False)
            f.write("\n")
    
    # Step 6: Compute question embeddings and similarity matrices
    print("\n🔢 Computing question embeddings and similarity matrices...")
    questions_data: list[QuestionData] = []
    
    # Build article lookup
    article_lookup = {a.article_id: a for a in articles_data}
    
    for i, qa in enumerate(all_qa_pairs):
        article = article_lookup.get(qa["article_id"])
        if not article:
            continue
        
        # Compute question embedding
        q_embedding = np.array(
            embed_model.get_text_embedding(qa["question"]),
            dtype=np.float32,
        )
        
        # Compute similarity to all sentences in article
        sentence_sims = compute_question_sentence_sims(q_embedding, article.embeddings)
        
        # Pre-compute top-K candidates
        top_k = min(args.top_k, len(sentence_sims))
        top_k_indices = np.argsort(sentence_sims)[::-1][:top_k].astype(np.int32)
        
        # Find answer sentence index
        answer_idx = find_answer_sentence_idx(article.sentences, qa["answer_sentence"])
        
        questions_data.append(QuestionData(
            question_id=i,
            article_id=qa["article_id"],
            question=qa["question"],
            answer_sentence=qa["answer_sentence"],
            answer_sentence_idx=answer_idx,
            embedding=q_embedding,
            sentence_sims=sentence_sims,
            top_k_indices=top_k_indices,
        ))
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(all_qa_pairs)} questions...")
    
    print(f"   ✅ Computed embeddings for {len(questions_data)} questions")
    
    # Step 7: Create and save corpus
    print(f"\n💾 Saving corpus cache to {corpus_path}...")
    corpus = CorpusData(
        articles=articles_data,
        questions=questions_data,
        embed_dim=embed_dim,
        top_k=args.top_k,
    )
    
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)
    
    # Summary
    elapsed = time.time() - start_time
    corpus_size_mb = corpus_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("✅ Corpus preparation complete!")
    print("=" * 60)
    print(f"   Articles: {len(articles_data)}")
    print(f"   Questions: {len(questions_data)}")
    print(f"   Embedding dim: {embed_dim}")
    print(f"   Top-K candidates: {args.top_k}")
    print(f"   Corpus size: {corpus_size_mb:.1f} MB")
    print(f"   Time elapsed: {elapsed:.1f}s")
    print()
    print(f"   📄 Articles: {articles_path}")
    print(f"   📄 Questions: {questions_path}")
    print(f"   📦 Corpus: {corpus_path}")


if __name__ == "__main__":
    asyncio.run(main())
