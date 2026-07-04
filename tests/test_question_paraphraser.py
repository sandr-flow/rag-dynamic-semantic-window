"""Tests for benchmark hardening via question paraphrasing (plan step П.2)."""

import asyncio

from src.answer_matching import content_token_overlap
from src.config import LLMProviderConfig
from src.question_paraphraser import paraphrase_question_async


def test_overlap_full_lexical_mirror():
    sentence = "The Treaty of Paris was signed in 1783 ending the war."
    question = "When was the Treaty of Paris signed?"
    assert content_token_overlap(question, sentence) == 1.0


def test_overlap_disjoint_vocabulary():
    sentence = "The Treaty of Paris was signed in 1783 ending the war."
    question = "Which year saw the formal conclusion of hostilities?"
    assert content_token_overlap(question, sentence) < 0.35


def test_overlap_ignores_stopwords_only_question():
    assert content_token_overlap("What is the of and?", "Anything at all.") == 0.0


def _run(coro):
    return asyncio.run(coro)


def _fake_config():
    return LLMProviderConfig(provider="custom", model="fake", base_url="http://fake")


def test_paraphrase_accepts_low_overlap_candidate(monkeypatch):
    sentence = "The Treaty of Paris was signed in 1783 ending the war."
    responses = iter([{"question": "Which year saw the formal conclusion of hostilities?"}])

    async def fake_chat(prompt, config, **kwargs):
        return next(responses)

    monkeypatch.setattr(
        "src.question_paraphraser.chat_completion_json_async", fake_chat
    )
    result = _run(
        paraphrase_question_async(
            "When was the Treaty of Paris signed?", "1783", sentence, _fake_config()
        )
    )
    assert result["accepted"]
    assert result["overlap"] <= 0.35
    assert result["original_overlap"] == 1.0
    assert "hostilities" in result["question"]


def test_paraphrase_retries_and_keeps_best(monkeypatch):
    sentence = "The Treaty of Paris was signed in 1783 ending the war."
    responses = iter(
        [
            {"question": "When was the Treaty of Paris signed and sealed?"},  # high overlap
            {"question": "Which agreement year ended the conflict formally?"},  # lower
            {"question": "Which year saw the formal conclusion of hostilities?"},  # low
        ]
    )
    prompts = []

    async def fake_chat(prompt, config, **kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(
        "src.question_paraphraser.chat_completion_json_async", fake_chat
    )
    result = _run(
        paraphrase_question_async(
            "When was the Treaty of Paris signed?", "1783", sentence, _fake_config()
        )
    )
    assert result["accepted"]
    # Forbidden-word feedback must appear in retry prompts.
    assert any("Do NOT use any of these words" in p for p in prompts)


def test_paraphrase_falls_back_to_original_on_errors(monkeypatch):
    async def fake_chat(prompt, config, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(
        "src.question_paraphraser.chat_completion_json_async", fake_chat
    )
    original = "When was the Treaty of Paris signed?"
    result = _run(
        paraphrase_question_async(
            original,
            "1783",
            "The Treaty of Paris was signed in 1783 ending the war.",
            _fake_config(),
        )
    )
    assert result["question"] == original
    assert not result["accepted"]


def test_hardened_qa_fields_survive_dataset_roundtrip(tmp_path, monkeypatch):
    from stand import artifacts, paths

    monkeypatch.setattr(paths, "DATASETS_DIR", tmp_path / "datasets")
    items = [
        {
            "title": "Doc",
            "text": "Alpha beta. Gamma delta.",
            "qa_pairs": [
                {
                    "question": "Which pair opens the text?",
                    "question_original": "What starts the doc?",
                    "paraphrase_overlap": 0.2,
                    "answer": "Alpha beta",
                    "answer_sentence": "Alpha beta.",
                }
            ],
        }
    ]
    artifacts.save_dataset("hard_mini", items, source="hardened:mini", qa_model="fake")
    loaded = artifacts.load_dataset_items("hard_mini")
    qa = loaded[0]["qa_pairs"][0]
    assert qa["question_original"] == "What starts the doc?"
    assert qa["paraphrase_overlap"] == 0.2
