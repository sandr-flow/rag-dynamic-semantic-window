"""Tests for the benchmark stand package (artifacts, config, runner)."""

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
    artifacts.save_tuned("ds", "mock", {"threshold": 0.9}, {"hit_rate": 0.8})
    assert artifacts.has_tuned("ds", "mock")
    tuned = artifacts.load_tuned("ds", "mock")
    assert tuned["params"]["threshold"] == 0.9


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
