"""Tests for cross-document extrahard pair generation and artifacts."""

from stand import artifacts
from stand.extrahard_pairs import build_cross_document_pairs


def _mini_items():
    return [
        {
            "id": 0,
            "title": "Alpha",
            "text": "Alpha beta gamma. Second sentence here.",
            "qa_pairs": [
                {"question": "What letter starts?", "answer": "Alpha", "answer_sentence": "Alpha beta gamma."},
                {"question": "What comes second?", "answer": "Second", "answer_sentence": "Second sentence here."},
            ],
        },
        {
            "id": 1,
            "title": "Beta",
            "text": "Delta epsilon zeta. Another line follows.",
            "qa_pairs": [
                {"question": "What greek letter?", "answer": "Delta", "answer_sentence": "Delta epsilon zeta."},
            ],
        },
        {
            "id": 2,
            "title": "Gamma",
            "text": "Omega is last. Prior words matter.",
            "qa_pairs": [
                {"question": "What is last?", "answer": "Omega", "answer_sentence": "Omega is last."},
            ],
        },
    ]


def test_cross_document_pairs_only(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts.paths, "DATASETS_DIR", tmp_path / "datasets")
    pairs = build_cross_document_pairs(_mini_items(), partners_per_question=2, seed=42)
    assert pairs
    for pair in pairs:
        assert pair["source_docs"][0] != pair["source_docs"][1]
        assert " and " in pair["question"]
        assert len(pair["answer_sentences"]) == 2
        assert pair["answer_sentences"][0] != pair["answer_sentences"][1]


def test_pair_sampling_is_reproducible_and_deduped():
    first = build_cross_document_pairs(_mini_items(), partners_per_question=2, seed=7)
    second = build_cross_document_pairs(_mini_items(), partners_per_question=2, seed=7)
    assert first == second

    questions = [p["question"] for p in first]
    assert len(questions) == len(set(questions))


def test_save_and_load_extrahard_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts.paths, "DATASETS_DIR", tmp_path / "datasets")

    source_items = _mini_items()
    artifacts.save_dataset("base", source_items, source="custom", qa_model="custom")

    pairs = build_cross_document_pairs(source_items, partners_per_question=1, seed=1)
    info = artifacts.save_extrahard_dataset(
        "extra",
        pairs,
        source_name="base",
        corpus_dataset="base",
        corpus_num_items=len(source_items),
        partners_per_question=1,
        pair_seed=1,
    )

    assert artifacts.is_extrahard("extra")
    assert info.kind == "extrahard"
    assert artifacts.load_corpus_items("extra") == artifacts.load_dataset_items("base")
    loaded_pairs = artifacts.load_eval_questions("extra")
    assert len(loaded_pairs) == len(pairs)
    assert loaded_pairs[0]["answer_sentences"]

    manifest = artifacts.load_manifest("extra")
    assert manifest["corpus_dataset"] == "base"
    assert (artifacts.dataset_dir("extra") / "pairs.jsonl").exists()
    assert not (artifacts.dataset_dir("extra") / "data.jsonl").exists()


def _tuning_items():
    """Documents long enough for the tune min-sentence floor."""
    sentences_a = [f"Alpha doc sentence {i} has enough content." for i in range(12)]
    sentences_b = [f"Beta doc sentence {i} has enough content." for i in range(12)]
    return [
        {
            "id": 0,
            "title": "Alpha",
            "text": " ".join(sentences_a),
            "qa_pairs": [
                {
                    "question": "Which alpha sentence is five?",
                    "answer": "five",
                    "answer_sentence": sentences_a[5],
                },
            ],
        },
        {
            "id": 1,
            "title": "Beta",
            "text": " ".join(sentences_b),
            "qa_pairs": [
                {
                    "question": "Which beta sentence is seven?",
                    "answer": "seven",
                    "answer_sentence": sentences_b[7],
                },
            ],
        },
    ]


def test_build_extrahard_tune_corpus():
    from stand.tune import _build_extrahard_corpus

    class RecordingEmbedding:
        def get_text_embedding_batch(self, texts, **kwargs):
            return [[float(len(text)), 1.0, 0.0] for text in texts]

    items = _tuning_items()
    pairs = build_cross_document_pairs(items, partners_per_question=1, seed=3)
    corpus = _build_extrahard_corpus(
        items,
        pairs,
        RecordingEmbedding(),
        source="extra",
        embedding_provider="mock",
        embedding_model="mock:3",
        phantom_window=0,
        top_k=5,
    )

    assert corpus.kind == "shared"
    assert len(corpus.articles) == 2
    assert len(corpus.global_sentences) == 24
    assert corpus.questions
    assert corpus.questions[0].source_article_ids
    assert all(
        len(q.sentence_sims) == len(corpus.global_sentences) for q in corpus.questions
    )


def test_extrahard_tune_split_uses_only_same_split_pairs():
    from src.corpus_data import question_is_valid, split_corpus_by_documents
    from stand.tune import _build_extrahard_corpus

    class RecordingEmbedding:
        def get_text_embedding_batch(self, texts, **kwargs):
            return [[float(len(text)), 1.0, 0.0] for text in texts]

    items = _tuning_items()
    pairs = build_cross_document_pairs(items, partners_per_question=1, seed=3)
    corpus = _build_extrahard_corpus(
        items,
        pairs,
        RecordingEmbedding(),
        source="extra",
        embedding_provider="mock",
        embedding_model="mock:3",
        phantom_window=0,
        top_k=5,
    )
    train, val = split_corpus_by_documents(corpus, train_ratio=0.5, seed=1)

    assert train.kind == "shared"
    assert val.kind == "shared"
    assert all(question_is_valid(q, train) for q in train.questions)
    assert all(question_is_valid(q, val) for q in val.questions)
    for question in train.questions:
        assert all(
            article_id in {article.article_id for article in train.articles}
            for article_id in question.source_article_ids
        )
