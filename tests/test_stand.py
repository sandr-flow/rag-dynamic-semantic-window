"""Tests for the benchmark stand package (artifacts, config, runner)."""

import numpy as np
import pytest

from src.tokens import count_tokens
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
    monkeypatch.setattr(paths, "EMBEDDING_CACHE_DIR", tmp_path / "artifacts" / "embedding_cache")
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
    assert config.index_mode == "shared"
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
    assert tuned["strategy"] == "dynamic_semantic"
    assert tuned["metrics_train"]["hit_rate"] == 0.9
    assert tuned["metrics_val"]["hit_rate"] == 0.8


def test_tuned_roundtrip_per_strategy(temp_artifacts):
    artifacts.save_tuned(
        "ds",
        "mock",
        {"chunk_size": 192, "chunk_overlap": 16},
        {"hit_rate": 0.7},
        strategy="naive",
    )
    artifacts.save_tuned(
        "ds",
        "mock",
        {"window_size": 2},
        {"hit_rate": 0.75},
        strategy="fixed_window",
    )
    assert artifacts.load_tuned("ds", "mock", strategy="naive")["params"]["chunk_size"] == 192
    assert artifacts.load_tuned("ds", "mock", strategy="fixed_window")["params"]["window_size"] == 2
    assert artifacts.load_tuned("ds", "mock") is None


def test_tune_corpus_uses_phantom_embedding_texts():
    from stand.tune import _build_corpus

    class RecordingEmbedding:
        def __init__(self):
            self.texts = []
            self.batch_calls = 0

        def get_text_embedding(self, text):
            self.texts.append(text)
            return [float(len(text)), 1.0, 0.0]

        def get_text_embedding_batch(self, texts, **kwargs):
            self.batch_calls += 1
            return [self.get_text_embedding(text) for text in texts]

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

    assert corpus.embedding_mode == "phantom_w1__adj_clean__dual_seed"
    assert corpus.phantom_window == 1
    assert embed_model.texts[0] == f"{sentences[0]} {sentences[1]}"
    assert embed_model.texts[5] == f"{sentences[4]} {sentences[5]} {sentences[6]}"
    assert corpus.questions[0].answer_sentence_idx == 5
    # Batched calls: phantom texts, clean sentences (adjacency), questions
    assert embed_model.batch_calls == 3


def test_tune_corpus_clean_adjacency_uses_second_batch():
    from stand.tune import _build_corpus, _neighbor_sims

    class RecordingEmbedding:
        def __init__(self):
            self.batches: list[list[str]] = []

        def get_text_embedding(self, text):
            return [float(len(text)), 1.0, 0.0]

        def get_text_embedding_batch(self, texts, **kwargs):
            self.batches.append(list(texts))
            return [self.get_text_embedding(text) for text in texts]

    sentences = [f"Sentence {i} has enough content padded {'x' * i}." for i in range(10)]
    item = {
        "title": "Doc",
        "text": " ".join(sentences),
        "qa_pairs": [
            {"question": "Which sentence?", "answer_sentence": sentences[5]},
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

    # Three batches: phantom texts, clean sentences, questions.
    assert len(embed_model.batches) == 3
    assert embed_model.batches[1] == sentences
    assert corpus.embedding_mode == "phantom_w1__adj_clean__dual_seed"
    assert corpus.adjacency_space == "clean"

    clean_matrix = np.array(
        [[float(len(s)), 1.0, 0.0] for s in sentences], dtype=np.float32
    )
    np.testing.assert_allclose(
        corpus.articles[0].neighbor_sims, _neighbor_sims(clean_matrix), rtol=1e-6
    )
    # Question sims still live in phantom space (matrix of phantom embeddings).
    phantom_matrix = np.array(
        [[float(len(t)), 1.0, 0.0] for t in embed_model.batches[0]], dtype=np.float32
    )
    np.testing.assert_allclose(corpus.articles[0].embeddings, phantom_matrix, rtol=1e-6)
    np.testing.assert_allclose(corpus.articles[0].clean_embeddings, clean_matrix, rtol=1e-6)
    # Dual-seed candidate list is the union of top-k from each space (≤ 2k).
    k = len(corpus.questions[0].top_k_indices)
    assert k >= 1
    assert k <= 2 * corpus.top_k


def test_seed_indices_union_clean_and_phantom():
    from stand.tune import _seed_indices

    phantom = np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 0.1]], dtype=np.float32)
    clean = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.1]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    sims, idx = _seed_indices(query, phantom, clean, k=1)
    assert sims.argmax() == 0
    assert list(idx) == [0, 1]


def test_dual_seed_retrieve_interleaves_clean_hits():
    from llama_index.core import Document, Settings
    from llama_index.core.embeddings import MockEmbedding

    from src.config import DynamicSemanticConfig
    from src.strategies import DynamicSemanticStrategy, _interleave_node_lists

    old = getattr(Settings, "_embed_model", None)
    Settings.embed_model = MockEmbedding(embed_dim=4)
    try:
        text = " ".join(
            f"Sentence number {i} talks about one shared topic here." for i in range(8)
        )
        strategy = DynamicSemanticStrategy(
            [Document(text=text)],
            top_k=2,
            dynamic_config=DynamicSemanticConfig(
                phantom_window=1, prefetch_multiplier=1, dual_seed=True
            ),
        )
        node_ids = list(strategy._clean_node_ids)
        n = len(node_ids)
        phantom = np.zeros((n, 4), dtype=np.float32)
        phantom[:, 0] = np.arange(1, n + 1, dtype=np.float32)
        phantom[:, 1] = 1.0
        clean = np.zeros((n, 4), dtype=np.float32)
        clean[:, 0] = np.arange(n, 0, -1, dtype=np.float32)
        clean[:, 1] = 1.0
        strategy._matrix = strategy._normalize_rows(phantom)
        strategy._matrix_node_ids = node_ids
        strategy._clean_matrix = strategy._normalize_rows(clean)
        query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        phantom_nodes = strategy._topk_from_matrix(
            strategy._matrix, strategy._matrix_node_ids, query_vec, 2
        )
        clean_nodes = strategy._topk_from_matrix(
            strategy._clean_matrix, strategy._clean_node_ids, query_vec, 2
        )
        merged = _interleave_node_lists(phantom_nodes, clean_nodes)
    finally:
        Settings._embed_model = old

    assert merged[0].node.node_id == node_ids[-1]
    assert merged[1].node.node_id == node_ids[0]


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

    rows = result["comparisons"]["rows"]
    assert {row["baseline"] for row in rows} == {"Naive Chunking"}
    assert {row["metric"] for row in rows} == {"hr@3", "mrr"}


def test_drop_questions_without_documents():
    from stand.runner import _drop_questions_without_documents

    corpus_items = [{"id": 0, "title": "kept"}, {"id": 2, "title": "also kept"}]
    questions = [
        {"question": "answered by an indexed doc", "source_docs": ["0"]},
        {"question": "answered by a dropped doc", "source_docs": ["1"]},
        {"question": "compound, one source dropped", "source_docs": ["2", "1"]},
        {"question": "compound, both indexed", "source_docs": ["0", "2"]},
        {"question": "provenance unknown"},
    ]

    kept = _drop_questions_without_documents(questions, corpus_items, verbose=False)

    assert [q["question"] for q in kept] == [
        "answered by an indexed doc",
        "compound, both indexed",
        # No source_docs means we cannot prove it is unanswerable; keep it.
        "provenance unknown",
    ]


@pytest.mark.parametrize("index_mode", ["per_document", "shared"])
def test_runner_skips_questions_of_unchunkable_documents(temp_artifacts, index_mode):
    """An unchunkable document takes its questions out of the metrics with it."""
    from stand.runner import run

    items = [
        {
            "title": "Chunkable",
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
        },
        {
            # One giant tokenless blob: exactly what PDF table extraction
            # produces, and what drop_unchunkable_items exists to remove.
            "title": "Unchunkable",
            "text": "filler " * 4000,
            "qa_pairs": [
                {"question": "Unanswerable once the doc is dropped?",
                 "answer_sentence": "filler filler filler"},
            ],
        },
    ]
    artifacts.save_dataset("mixed", items, source="custom", qa_model="custom")

    result = run(
        RunConfig(
            dataset="mixed",
            embedding="mock",
            strategies=["naive"],
            index_mode=index_mode,
            top_k=3,
        ),
        verbose=False,
    )

    assert result["dataset"]["docs"] == 1
    assert result["dataset"]["questions"] == 1


def test_runner_second_run_hits_embedding_cache(temp_artifacts):
    from llama_index.core import Settings

    from stand.runner import run

    items = [
        {
            "title": "Cache",
            "text": (
                "Caching stores computed values for reuse. "
                "A cache hit avoids recomputation entirely. "
                "A cache miss falls through to the model. "
                "Eviction keeps the store bounded."
            ),
            "qa_pairs": [
                {"question": "What does a cache hit avoid?",
                 "answer_sentence": "A cache hit avoids recomputation entirely."},
            ],
        }
    ]
    artifacts.save_dataset("mini_cache", items, source="custom", qa_model="custom")
    config = RunConfig(
        dataset="mini_cache",
        embedding="mock",
        strategies=["naive", "dynamic_semantic"],
        top_k=3,
    )

    run(config, verbose=False)
    store_dirs = [
        p for p in (temp_artifacts / "artifacts" / "embedding_cache").iterdir() if p.is_dir()
    ]
    assert len(store_dirs) == 1
    assert list(store_dirs[0].glob("shard-*.npy"))

    # Second run builds a fresh store over the same shards: everything must hit.
    run(config, verbose=False)
    stats = Settings.embed_model.cache_stats
    assert stats["misses"] == 0
    assert stats["hits"] > 0


def test_resolve_tuned_from_other_dataset(temp_artifacts):
    from stand.runner import _resolve_tuned

    artifacts.save_tuned(
        "extrahard_src",
        "mock",
        {"threshold": 0.42, "phantom_window": 1},
        {"hit_rate": 0.5},
    )
    config = RunConfig(
        dataset="wiki_hard",
        embedding="mock",
        params="tuned",
        tuned_dataset="extrahard_src",
    )
    params, tuned = _resolve_tuned(config)
    assert params["threshold"] == 0.42
    assert tuned["dataset"] == "extrahard_src"


def test_runner_rejects_unknown_dynamic_overrides(temp_artifacts):
    from stand.runner import _resolve_tuned

    config = RunConfig(
        dataset="whatever",
        dynamic_overrides={"min_window": 0, "bogus_knob": 1},
    )
    with pytest.raises(ValueError, match="bogus_knob"):
        _resolve_tuned(config)


def test_runner_seeds_only_overrides(temp_artifacts):
    """min_window=0 + max_expand=0 degenerates dynamic clusters to seeds."""
    from stand.runner import run

    items = [
        {
            "title": "Ablate",
            "text": (
                "Alpha topic sentence about databases. "
                "Beta topic sentence about indexing. "
                "Gamma topic sentence about caching. "
                "Delta topic sentence about sharding."
            ),
            "qa_pairs": [
                {"question": "What is said about caching?",
                 "answer_sentence": "Gamma topic sentence about caching."},
            ],
        }
    ]
    artifacts.save_dataset("mini_ablate", items, source="custom", qa_model="custom")

    overrides = {"min_window": 0, "max_expand": 0, "merge_gap": 0}
    config = RunConfig(
        dataset="mini_ablate",
        embedding="mock",
        strategies=["dynamic_semantic"],
        top_k=3,
        dynamic_overrides=overrides,
    )
    result = run(config, verbose=False)

    assert result["config"]["dynamic_overrides"] == overrides
    # Every retrieved unit is a single sentence: tokens per question must be
    # well below the full 4-sentence document.
    row = result["summary"][0]
    assert row["strategy"] == "Dynamic Semantic"
    full_doc_tokens = count_tokens(items[0]["text"])
    assert 0 < row["tokens"] < full_doc_tokens


def test_runner_extrahard_requires_shared_index(temp_artifacts):
    from stand.extrahard_pairs import build_cross_document_pairs
    from stand.runner import run

    items = [
        {
            "id": 0,
            "title": "One",
            "text": "Alpha beta. Second line.",
            "qa_pairs": [
                {"question": "Q one?", "answer": "Alpha", "answer_sentence": "Alpha beta."},
            ],
        },
        {
            "id": 1,
            "title": "Two",
            "text": "Gamma delta. Fourth line.",
            "qa_pairs": [
                {"question": "Q two?", "answer": "Gamma", "answer_sentence": "Gamma delta."},
            ],
        },
    ]
    artifacts.save_dataset("corp", items, source="custom", qa_model="custom")
    pairs = build_cross_document_pairs(items, partners_per_question=1, seed=0)
    artifacts.save_extrahard_dataset(
        "extra",
        pairs,
        source_name="corp",
        corpus_dataset="corp",
        corpus_num_items=len(items),
        partners_per_question=1,
        pair_seed=0,
    )

    config = RunConfig(
        dataset="extra",
        embedding="mock",
        strategies=["naive"],
        index_mode="per_document",
        top_k=2,
    )
    with pytest.raises(ValueError, match="requires --index-mode shared"):
        run(config, verbose=False)

    result = run(
        RunConfig(
            dataset="extra",
            embedding="mock",
            strategies=["naive"],
            index_mode="shared",
            top_k=2,
        ),
        verbose=False,
    )
    assert result["dataset"]["questions"] == len(pairs)
    assert "partial_hr@2" in result["summary"][0]


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


def _three_doc_items():
    return [
        {
            "id": i,
            "title": f"Doc {i}",
            "text": (
                f"Topic {i} opens with a clear statement of the claim. "
                f"Supporting evidence for topic {i} follows in the next sentence. "
                f"A third sentence adds background about topic {i}. "
                f"The closing sentence restates the answer about topic {i}."
            ),
            "qa_pairs": [
                {
                    "question": f"What is the claim in topic {i}?",
                    "answer_sentence": f"Topic {i} opens with a clear statement of the claim.",
                }
            ],
        }
        for i in range(3)
    ]


def test_split_items_keeps_compound_questions_inside_one_split():
    from stand.tune_live import split_items_by_documents

    items = [{"id": "a", "text": "A."}, {"id": "b", "text": "B."}, {"id": "c", "text": "C."}]
    questions = [
        {"question": "a only", "source_docs": ["a"]},
        {"question": "b+c", "source_docs": ["b", "c"]},
        {"question": "a+c", "source_docs": ["a", "c"]},
    ]
    train_items, train_q, val_items, val_q = split_items_by_documents(
        items, questions, train_ratio=0.67, seed=0
    )
    train_ids = {str(item["id"]) for item in train_items}
    val_ids = {str(item["id"]) for item in val_items}
    assert train_ids.isdisjoint(val_ids)
    for q in train_q:
        assert all(doc in train_ids for doc in q["source_docs"])
    for q in val_q:
        assert all(doc in val_ids for doc in q["source_docs"])


def test_runner_applies_per_strategy_tuned_params(temp_artifacts):
    from stand.runner import run

    artifacts.save_dataset("mini_tuned", _three_doc_items(), source="custom", qa_model="custom")
    artifacts.save_tuned(
        "mini_tuned",
        "mock",
        {"chunk_size": 128, "chunk_overlap": 0},
        {"hit_rate": 0.5},
        strategy="naive",
    )
    artifacts.save_tuned(
        "mini_tuned",
        "mock",
        {"threshold": 0.5, "min_window": 0, "max_expand": 1},
        {"hit_rate": 0.5},
        strategy="dynamic_semantic",
    )

    result = run(
        RunConfig(
            dataset="mini_tuned",
            embedding="mock",
            strategies=["naive", "dynamic_semantic"],
            params="tuned",
            top_k=3,
        ),
        verbose=False,
    )
    assert result["config"]["strategy_overrides"]["naive"]["chunk_size"] == 128
    assert result["config"]["strategy_overrides"]["dynamic_semantic"]["threshold"] == 0.5
    assert set(result["tuned_by_strategy"]) == {"naive", "dynamic_semantic"}


def test_tune_live_naive_mock(temp_artifacts):
    from stand.tune import tune

    artifacts.save_dataset("tune_mini", _three_doc_items(), source="custom", qa_model="custom")
    out = tune(
        "tune_mini",
        "mock",
        strategy="naive",
        n_trials=2,
        index_mode="shared",
        soft_token_limit=5000,
        split_seed=42,
    )
    assert out["strategy"] == "naive"
    assert "chunk_size" in out["params"]
    assert artifacts.has_tuned("tune_mini", "mock", strategy="naive")
    loaded = artifacts.load_tuned("tune_mini", "mock", strategy="naive")
    assert loaded["tuning"]["path"] == "live"
