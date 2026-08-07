"""FinanceBench loader tests (offline: network and PDF extraction mocked)."""

import json

from src import financebench_loader as fbl
from src.answer_matching import normalize_answer_text


def test_evidence_texts_tolerates_schemas():
    assert fbl._evidence_texts({"evidence": "plain string"}) == ["plain string"]
    assert fbl._evidence_texts({"evidence": ["a", "b"]}) == ["a", "b"]
    assert fbl._evidence_texts(
        {"evidence": [{"evidence_text": "from dict"}, {"other": "x"}]}
    ) == ["from dict"]
    assert fbl._evidence_texts({}) == []


def test_resolve_answer_sentence_prefers_sentence_found_in_doc():
    doc = (
        "The company reported total revenue of $12.3 billion for fiscal 2022, "
        "an increase of 8% year over year. Operating margin declined slightly."
    )
    doc_norm = normalize_answer_text(doc)
    evidence = [
        "Short cell text. "
        "The company reported total revenue of $12.3 billion for fiscal 2022, "
        "an increase of 8% year over year."
    ]
    resolved = fbl.resolve_answer_sentence(evidence, doc_norm)
    assert "total revenue of $12.3 billion" in resolved
    assert normalize_answer_text(resolved) in doc_norm


def test_resolve_answer_sentence_falls_back_to_longest():
    evidence = [
        "This distinctive evidence sentence does not appear in the document at all, "
        "but it is long enough to serve as a fuzzy anchor."
    ]
    resolved = fbl.resolve_answer_sentence(evidence, normalize_answer_text("unrelated text"))
    assert resolved == evidence[0]


def test_load_financebench_items_groups_by_doc(monkeypatch, tmp_path):
    records = [
        {
            "financebench_id": "fb_1",
            "doc_name": "ACME_2022_10K",
            "question": "What was ACME's 2022 revenue?",
            "answer": "$12.3 billion",
            "evidence": [
                {
                    "evidence_text": (
                        "ACME reported total revenue of $12.3 billion for fiscal 2022, "
                        "up 8% from the prior year."
                    )
                }
            ],
        },
        {
            "financebench_id": "fb_2",
            "doc_name": "ACME_2022_10K",
            "question": "Did operating margin improve in 2022?",
            "answer": "No",
            "evidence": [
                {
                    "evidence_text": (
                        "Operating margin declined to 18.2% in fiscal 2022 "
                        "from 19.1% in fiscal 2021."
                    )
                }
            ],
        },
        {
            "financebench_id": "fb_3",
            "doc_name": "OTHER_2021_10K",
            "question": "Unused when max_docs=1?",
            "answer": "n/a",
            "evidence": [{"evidence_text": "Something long enough to pass the length filter."}],
        },
    ]
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )

    doc_text = (
        "ACME Corporation Annual Report. "
        "ACME reported total revenue of $12.3 billion for fiscal 2022, up 8% from "
        "the prior year. Operating margin declined to 18.2% in fiscal 2022 from "
        "19.1% in fiscal 2021. Forward-looking statements follow."
    )

    def fake_download(relative_path, cache_dir, *, binary=False):
        if relative_path == fbl.QUESTIONS_FILE:
            return questions_path
        return tmp_path / relative_path.replace("/", "_")

    monkeypatch.setattr(fbl, "_download", fake_download)
    monkeypatch.setattr(fbl, "extract_pdf_text", lambda path: doc_text)

    items = fbl.load_financebench_items(cache_dir=tmp_path, max_docs=1)

    assert len(items) == 1
    item = items[0]
    assert item["title"] == "ACME_2022_10K"
    assert item["text"] == doc_text
    assert [qa["financebench_id"] for qa in item["qa_pairs"]] == ["fb_1", "fb_2"]
    for qa in item["qa_pairs"]:
        assert normalize_answer_text(qa["answer_sentence"]) in normalize_answer_text(doc_text)
