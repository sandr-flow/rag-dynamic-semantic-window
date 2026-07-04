"""Tests for the benchmark stand package (artifacts, config, runner)."""

import numpy as np
import pytest

from stand import artifacts, paths
from stand.artifacts import EmbeddingInfo
from stand.runconfig import RunConfig


@pytest.fixture
def temp_artifacts(tmp_path, monkeypatch):
    """Redirect all artifact/result paths into a temp directory."""
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "DATASETS_DIR", tmp_path / "artifacts" / "datasets")
    monkeypatch.setattr(paths, "TUNED_DIR", tmp_path / "artifacts" / "tuned")
    monkeypatch.setattr(paths, "CORPUS_CACHE_DIR", tmp_path / "artifacts" / "corpus_cache")
    monkeypatch.setattr(paths, "EMBEDDINGS_REGISTRY", tmp_path / "artifacts" / "embeddings.json")
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    paths.ensure_dirs()
    return tmp_path


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


def test_runconfig_defaults_and_metric_k():
    config = RunConfig(dataset="d", top_k=7)
    assert config.embedding == "mock"
    assert config.index_mode == "per_document"
    assert config.effective_metric_k == 7
    assert "dynamic_semantic" in config.strategies


def test_runconfig_rejects_bad_values():
    with pytest.raises(ValueError):
        RunConfig(dataset="d", index_mode="nonsense")
    with pytest.raises(ValueError):
        RunConfig(dataset="d", params="magic")
    with pytest.raises(ValueError):
        RunConfig(dataset="d", strategies=[])


def test_runconfig_from_dict_ignores_unknown_keys():
    config = RunConfig.from_dict({"dataset": "d", "index_mode": "shared", "bogus": 1})
    assert config.index_mode == "shared"


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_slugify():
    assert artifacts.slugify("QASPER Val 2024!") == "qasper_val_2024"
    assert artifacts.slugify("  ") == "unnamed"


def test_dataset_roundtrip(temp_artifacts):
    items = [
        {
            "title": "Doc",
            "text": "Alpha beta. Gamma delta.",
            "qa_pairs": [{"question": "q?", "answer_sentence": "Alpha beta."}],
        }
    ]
    info = artifacts.save_dataset("My Set", items, source="custom", qa_model="custom")

    assert info.name == "my_set"
    assert info.num_items == 1
    assert info.num_questions == 1

    listed = artifacts.list_datasets()
    assert [d.name for d in listed] == ["my_set"]

    loaded = artifacts.load_dataset_items("My Set")
    assert loaded[0]["qa_pairs"][0]["answer_sentence"] == "Alpha beta."


def test_embedding_registry_builtins_and_register(temp_artifacts):
    names = {info.name for info in artifacts.list_embeddings()}
    assert {"mock", "bge-small"} <= names

    artifacts.register_embedding(
        EmbeddingInfo(name="my-openai", provider="openai", model="text-embedding-3-small")
    )
    info = artifacts.get_embedding("my-openai")
    assert info is not None
    assert info.provider == "openai"


def test_tuned_roundtrip(temp_artifacts):
    assert not artifacts.has_tuned("ds", "mock")
    artifacts.save_tuned(
        "ds",
        "mock",
        {"threshold": 0.9},
        {"hit_rate": 0.8},
        metrics_train={"hit_rate": 0.9},
        metrics_val={"hit_rate": 0.8},
    )
    assert artifacts.has_tuned("ds", "mock")
    tuned = artifacts.load_tuned("ds", "mock")
    assert tuned["params"]["threshold"] == 0.9
    assert tuned["metrics_train"]["hit_rate"] == 0.9
    assert tuned["metrics_val"]["hit_rate"] == 0.8


def test_tune_corpus_uses_phantom_embedding_texts():
    from stand.tune import _build_corpus

    class RecordingEmbedding:
        def __init__(self):
            self.texts = []

        def get_text_embedding(self, text):
            self.texts.append(text)
            return [float(len(text)), 1.0, 0.0]

    sentences = [f"Sentence {i} has enough content." for i in range(10)]
    item = {
        "title": "Doc",
        "text": " ".join(sentences),
        "qa_pairs": [
            {
                "question": "Which sentence is number five?",
                "answer_sentence": sentences[5],
            }
        ],
    }
    embed_model = RecordingEmbedding()

    corpus = _build_corpus(
        [item],
        embed_model,
        source="mini",
        embedding_provider="mock",
        embedding_model="mock:3",
        phantom_window=1,
    )

    assert corpus.embedding_mode == "phantom_w1"
    assert corpus.phantom_window == 1
    assert embed_model.texts[0] == f"{sentences[0]} {sentences[1]}"
    assert embed_model.texts[5] == f"{sentences[4]} {sentences[5]} {sentences[6]}"
    assert corpus.questions[0].answer_sentence_idx == 5


def test_corpus_document_split_keeps_article_boundaries():
    from src.corpus_data import ArticleData, CorpusData, QuestionData, split_corpus_by_documents

    articles = [
        ArticleData(i, f"doc_{i}", [f"sentence {i}"], np.ones((1, 2)), np.array([]))
        for i in range(5)
    ]
    questions = [
        QuestionData(i, i, "q", "a", 0, np.ones(2), np.ones(1), np.array([0]))
        for i in range(5)
    ]
    corpus = CorpusData(articles=articles, questions=questions)

    train, val = split_corpus_by_documents(corpus, train_ratio=0.6, seed=7)

    train_ids = {article.article_id for article in train.articles}
    val_ids = {article.article_id for article in val.articles}
    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)
    assert {question.article_id for question in train.questions} == train_ids
    assert {question.article_id for question in val.questions} == val_ids


# ---------------------------------------------------------------------------
# Runner (end to end with offline mock embeddings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_mode", ["per_document", "shared"])
def test_runner_end_to_end_mock(temp_artifacts, index_mode):
    from stand.runner import run

    items = [
        {
            "title": "Quantum",
            "text": (
                "Quantum mechanics describes nature at small scales. "
                "Superposition lets a system exist in multiple states. "
                "Entanglement correlates distant particles. "
                "Measurement collapses the wavefunction."
            ),
            "qa_pairs": [
                {"question": "What is superposition?",
                 "answer_sentence": "Superposition lets a system exist in multiple states."},
            ],
        }
    ]
    artifacts.save_dataset("mini", items, source="custom", qa_model="custom")

    config = RunConfig(
        dataset="mini",
        embedding="mock",
        strategies=["naive", "dynamic_semantic"],
        index_mode=index_mode,
        top_k=3,
    )
    result = run(config, verbose=False)

    names = {row["strategy"] for row in result["summary"]}
    assert {"Naive Chunking", "Dynamic Semantic"} == names
    assert result["dataset"]["questions"] == 1
    assert "result_path" in result


def test_document_metadata_is_excluded_from_embed_content():
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import MetadataMode

    from stand.runner import _document_from_item

    item = {"id": 3, "title": "Some Long Article Title",
            "text": "First sentence here. Second sentence here."}
    document = _document_from_item(item, 0)
    nodes = SentenceSplitter(chunk_size=64, chunk_overlap=0).get_nodes_from_documents(
        [document]
    )

    embed_content = nodes[0].get_content(metadata_mode=MetadataMode.EMBED)
    assert "source_doc" not in embed_content
    assert "Some Long Article Title" not in embed_content
    assert nodes[0].metadata["source_doc"] == "3"
