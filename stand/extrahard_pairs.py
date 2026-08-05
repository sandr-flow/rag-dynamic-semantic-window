"""Cross-document compound question generation for extrahard datasets."""

from __future__ import annotations

import random
from typing import Any

MAX_COMPOUND_QUESTION_CHARS = 2000


def _qa_key(doc_id: str, qa_index: int) -> tuple[str, int]:
    return (doc_id, qa_index)


def _unordered_pair_key(left: dict[str, Any], right: dict[str, Any]) -> tuple:
    a = _qa_key(left["doc_id"], left["qa_index"])
    b = _qa_key(right["doc_id"], right["qa_index"])
    return (a, b) if a <= b else (b, a)


def build_question_pool(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten corpus items into per-question records with stable doc ids."""
    pool: list[dict[str, Any]] = []
    for item_idx, item in enumerate(items):
        doc_id = str(item.get("id", item_idx))
        title = item.get("title", f"item_{item_idx}")
        for qa_index, qa in enumerate(item.get("qa_pairs", [])):
            pool.append(
                {
                    "doc_id": doc_id,
                    "qa_index": qa_index,
                    "title": title,
                    "question": qa["question"],
                    "answer": qa.get("answer", ""),
                    "answer_sentence": qa.get("answer_sentence", qa.get("answer", "")),
                }
            )
    return pool


def build_cross_document_pairs(
    items: list[dict[str, Any]],
    *,
    partners_per_question: int = 2,
    seed: int = 42,
    max_question_chars: int = MAX_COMPOUND_QUESTION_CHARS,
) -> list[dict[str, Any]]:
    """Combine hard questions from different documents into compound queries."""
    pool = build_question_pool(items)
    if len(pool) < 2:
        return []

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for entry in pool:
        by_doc.setdefault(entry["doc_id"], []).append(entry)

    rng = random.Random(seed)
    seen: set[tuple] = set()
    pairs: list[dict[str, Any]] = []

    for entry in pool:
        candidates = [e for e in pool if e["doc_id"] != entry["doc_id"]]
        if len(candidates) < partners_per_question:
            continue
        partners = rng.sample(candidates, partners_per_question)
        for partner in partners:
            pair_key = _unordered_pair_key(entry, partner)
            if pair_key in seen:
                continue
            if entry["answer_sentence"] == partner["answer_sentence"]:
                continue

            question = f"{entry['question']} and {partner['question']}"
            if len(question) >= max_question_chars:
                continue

            seen.add(pair_key)
            pairs.append(
                {
                    "question": question,
                    "question_parts": [entry["question"], partner["question"]],
                    "answer_sentences": [
                        entry["answer_sentence"],
                        partner["answer_sentence"],
                    ],
                    "answers": [entry["answer"], partner["answer"]],
                    "source_docs": [entry["doc_id"], partner["doc_id"]],
                    "source_titles": [entry["title"], partner["title"]],
                    "source_question_ids": [entry["qa_index"], partner["qa_index"]],
                }
            )

    return pairs
