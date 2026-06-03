"""Convert failures to Optuna corpus format.

Creates a cached corpus from the failures directory for targeted optimization.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.corpus_data import ArticleData, CorpusData, QuestionData, find_answer_sentence_idx
from src.utils import split_into_sentences

load_dotenv()


def compute_neighbor_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between adjacent sentences."""
    if len(embeddings) < 2:
        return np.array([])
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    
    neighbor_sims = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return neighbor_sims


def compute_question_sentence_sims(
    query_embedding: np.ndarray,
    sentence_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute similarity between query and all sentences."""
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return np.zeros(len(sentence_embeddings))
    
    query_normalized = query_embedding / query_norm
    
    sent_norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sent_norms = np.where(sent_norms == 0, 1, sent_norms)
    sent_normalized = sentence_embeddings / sent_norms
    
    return np.dot(sent_normalized, query_normalized)


def load_failures() -> list[dict]:
    """Load failures - only the failed question per file."""
    failures_dir = Path(__file__).parent / "results" / "failures"
    
    failures = []
    
    for failure_file in failures_dir.glob("*.json"):
        try:
            with open(failure_file, "r", encoding="utf-8") as f:
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
    """Create Optuna corpus from failures."""
    print("=" * 60)
    print("Create Optuna Corpus from Failures")
    print("=" * 60)
    
    # Load embedding model
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"\nLoading: {model_name}")
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.embed_model = embed_model
    
    # Load failures
    print("\n📥 Loading failures...")
    failures = load_failures()
    print(f"   Loaded {len(failures)} failed questions")
    
    # Group by article text
    articles_map = {}
    for f in failures:
        key = hash(f["text"][:1000])
        if key not in articles_map:
            articles_map[key] = {
                "title": f["title"],
                "text": f["text"],
                "questions": [],
            }
        articles_map[key]["questions"].append({
            "question": f["question"],
            "answer": f["answer"],
        })
    
    articles_list = list(articles_map.values())
    print(f"   Unique articles: {len(articles_list)}")
    
    # Process articles
    print("\n🔧 Processing articles...")
    articles = []
    questions = []
    
    for i, article in enumerate(articles_list):
        print(f"  [{i+1}/{len(articles_list)}] {article['title'][:50]}...")
        
        # Split into sentences
        sentences = split_into_sentences(article["text"])
        if len(sentences) < 5:
            continue
        
        # Compute embeddings
        sentence_embeddings = np.array([
            embed_model.get_text_embedding(s) for s in sentences
        ], dtype=np.float32)
        
        # Compute neighbor similarities
        neighbor_sims = compute_neighbor_similarities(sentence_embeddings)
        
        article_idx = len(articles)
        
        article_data = ArticleData(
            article_id=article_idx,
            title=article["title"],
            sentences=sentences,
            embeddings=sentence_embeddings,
            neighbor_sims=neighbor_sims,
        )
        articles.append(article_data)
        
        # Process questions
        for qa in article["questions"]:
            query_embedding = np.array(
                embed_model.get_query_embedding(qa["question"]),
                dtype=np.float32
            )
            
            sentence_sims = compute_question_sentence_sims(
                query_embedding, sentence_embeddings
            )
            
            # Get top-k indices
            k = min(100, len(sentences))
            top_k_indices = np.argsort(sentence_sims)[::-1][:k]
            
            # Find answer sentence
            answer_idx = find_answer_sentence_idx(sentences, qa["answer"])
            
            question_data = QuestionData(
                question_id=len(questions),
                article_id=article_idx,
                question=qa["question"],
                answer_sentence=qa["answer"],
                answer_sentence_idx=answer_idx,
                embedding=query_embedding,
                sentence_sims=sentence_sims,
                top_k_indices=top_k_indices,
            )
            questions.append(question_data)
    
    # Create corpus
    corpus = CorpusData(articles=articles, questions=questions)
    
    # Save
    output_path = Path(__file__).parent / "data" / "failures_corpus.pkl"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(corpus, f)
    
    valid_q = sum(1 for q in questions if q.answer_sentence_idx >= 0)
    
    print(f"\n✅ Corpus saved to: {output_path}")
    print(f"   Articles: {len(articles)}")
    print(f"   Questions: {len(questions)}")
    print(f"   Valid (answer found): {valid_q}/{len(questions)}")


if __name__ == "__main__":
    main()
