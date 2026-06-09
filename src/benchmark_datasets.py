"""Benchmark dataset loaders.

Supported formats:
- Combined JSON/JSONL articles with qa_pairs:
  {"title": "...", "text": "...", "qa_pairs": [{"question": "...", "answer_sentence": "..."}]}
- JSON object with {"articles": [...], "questions": [...]}.
- Separate articles/questions JSONL files compatible with prepare_corpus.py output.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BenchmarkItem = dict[str, Any]


def load_benchmark_dataset(
    dataset_path: str | None = None,
    articles_path: str | None = None,
    questions_path: str | None = None,
) -> list[BenchmarkItem]:
    """Load benchmark items from combined or paired dataset files."""
    if dataset_path:
        payload = _load_json_or_jsonl(Path(dataset_path))
        return _normalize_combined_payload(payload)

    if articles_path and questions_path:
        articles = _load_json_or_jsonl(Path(articles_path))
        questions = _load_json_or_jsonl(Path(questions_path))
        return _join_articles_and_questions(articles, questions)

    raise ValueError("Provide either dataset_path or both articles_path and questions_path")


def _load_json_or_jsonl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _normalize_combined_payload(payload: Any) -> list[BenchmarkItem]:
    if isinstance(payload, dict):
        if "articles" in payload and "questions" in payload:
            return _join_articles_and_questions(payload["articles"], payload["questions"])
        if "items" in payload:
            payload = payload["items"]
        else:
            payload = [payload]

    if not isinstance(payload, list):
        raise ValueError("Combined benchmark dataset must be a list or object")

    items = []
    for idx, record in enumerate(payload):
        item = _normalize_article_record(record, fallback_id=idx)
        if item["qa_pairs"]:
            items.append(item)

    if not items:
        raise ValueError("Benchmark dataset contains no items with qa_pairs")
    return items


def _join_articles_and_questions(articles_payload: Any, questions_payload: Any) -> list[BenchmarkItem]:
    articles = _as_records(articles_payload, key="articles")
    questions = _as_records(questions_payload, key="questions")

    questions_by_article = defaultdict(list)
    for question in questions:
        article_id = question.get("article_id", question.get("article"))
        if article_id is None:
            raise ValueError(f"Question is missing article_id: {question}")
        questions_by_article[str(article_id)].append(_normalize_qa_pair(question))

    items = []
    for idx, article in enumerate(articles):
        item = _normalize_article_record(article, fallback_id=idx)
        article_id = article.get("id", article.get("article_id", idx))
        item["qa_pairs"] = questions_by_article.get(str(article_id), [])
        if item["qa_pairs"]:
            items.append(item)

    if not items:
        raise ValueError("No article/question pairs matched")
    return items


def _normalize_article_record(record: dict[str, Any], fallback_id: int) -> BenchmarkItem:
    if not isinstance(record, dict):
        raise ValueError(f"Article record must be an object: {record}")

    text = record.get("text") or record.get("content") or record.get("document")
    if not text:
        raise ValueError(f"Article record is missing text/content/document: {record}")

    qa_pairs = record.get("qa_pairs") or record.get("questions") or []
    return {
        "id": record.get("id", record.get("article_id", fallback_id)),
        "title": record.get("title", record.get("article_title", f"item_{fallback_id}")),
        "text": text,
        "qa_pairs": [_normalize_qa_pair(qa) for qa in qa_pairs],
    }


def _normalize_qa_pair(record: dict[str, Any]) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError(f"QA record must be an object: {record}")

    question = record.get("question") or record.get("query")
    answer_sentence = (
        record.get("answer_sentence")
        or record.get("reference")
        or record.get("expected_answer")
        or record.get("answer")
    )
    answer = record.get("answer", answer_sentence)

    if not question or not answer_sentence:
        raise ValueError(f"QA record must include question and answer_sentence/answer: {record}")

    return {
        "question": str(question),
        "answer": str(answer),
        "answer_sentence": str(answer_sentence),
    }


def _as_records(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get(key, payload.get("items"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list of {key}")
    return payload
