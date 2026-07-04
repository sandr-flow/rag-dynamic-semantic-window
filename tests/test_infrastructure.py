import json
import subprocess
import sys

from run_optuna import ensure_sqlite_storage_parent
from src.benchmark_datasets import load_benchmark_dataset
from src.corpus_data import find_answer_sentence_idx
from src.hpo_config import load_hpo_settings
from src.metrics import hit_rate
from src.providers import (
    build_embedding_model,
    embedding_config_from_env,
    llm_config_from_env,
    provider_catalog,
)
from src.result_summary import build_leaderboard, summarize_benchmark_file
from src.strategy_registry import (
    default_strategy_params,
    parse_strategy_ids,
    resolve_strategy_overrides,
    strategy_catalog,
    validate_strategy_overrides_for_ids,
)
from src.utils import split_into_sentences


def test_parse_strategy_aliases():
    assert parse_strategy_ids("sentence,dynamic") == ["naive", "dynamic_semantic"]
    assert "token_text" in parse_strategy_ids("default")
    # Only text strategies remain, so "all" == "default".
    assert set(parse_strategy_ids("all")) == set(parse_strategy_ids("default"))
    assert "code" not in parse_strategy_ids("all")
    assert parse_strategy_ids("sentence_splitter,token-text") == ["naive", "token_text"]


def test_strategy_and_provider_catalogs_include_core_entries():
    strategies = {item["id"]: item for item in strategy_catalog()}
    providers = provider_catalog()

    assert strategies["dynamic_semantic"]["default"] is True
    assert "threshold" in strategies["dynamic_semantic"]["override_keys"]
    assert strategies["dynamic_semantic"]["default_params"]["threshold"] == default_strategy_params(
        "dynamic"
    )["threshold"]
    assert "chunk_size" in strategies["token_text"]["default_params"]
    assert "buffer_size" in strategies["semantic_splitter"]["default_params"]
    assert "html" not in strategies and "code" not in strategies
    assert any(item["id"] == "openrouter" for item in providers["llms"])
    assert any(item["id"] == "mock" for item in providers["embeddings"])
    assert any(item["id"] == "custom" for item in providers["embeddings"])


def test_provider_check_dry_run_and_mock_embedding():
    dry_run = subprocess.run(
        [
            sys.executable,
            "check_providers.py",
            "--embedding-provider",
            "mock",
            "--embedding-model",
            "mock:12",
            "--llm-provider",
            "openrouter",
            "--llm-model",
            "openai/gpt-4.1-mini",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    live_embedding = subprocess.run(
        [
            sys.executable,
            "check_providers.py",
            "--run",
            "--skip-llm",
            "--embedding-provider",
            "mock",
            "--embedding-model",
            "mock:12",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Provider dry-run completed" in dry_run.stdout
    assert "openrouter" in dry_run.stdout
    assert "Embedding request returned 12 dimensions" in live_embedding.stdout


def test_resolve_flat_and_grouped_overrides():
    flat = {"threshold": 0.9, "max_expand": 4}
    assert resolve_strategy_overrides("dynamic", flat) == flat
    assert resolve_strategy_overrides("naive", flat) == {}

    grouped = {
        "global": {"chunk_size": 128},
        "dynamic_semantic": {"threshold": 0.91},
        "naive": {"chunk_overlap": 12},
    }
    assert resolve_strategy_overrides("dynamic_semantic", grouped) == {
        "chunk_size": 128,
        "threshold": 0.91,
    }
    assert resolve_strategy_overrides("naive", grouped) == {
        "chunk_size": 128,
        "chunk_overlap": 12,
    }

    hyphen_grouped = {"dynamic-semantic": {"threshold": 0.92}}
    assert resolve_strategy_overrides("dynamic_semantic", hyphen_grouped) == {"threshold": 0.92}


def test_validate_strategy_overrides_catches_unknown_keys():
    assert validate_strategy_overrides_for_ids(
        ["dynamic_semantic"],
        {"threshold": 0.9, "max_expand": 4},
    ) == []
    assert validate_strategy_overrides_for_ids(
        ["naive", "dynamic_semantic"],
        {"global": {"chunk_size": 128}, "dynamic_semantic": {"threshold": 0.9}},
    ) == ["dynamic_semantic received unsupported override keys: chunk_size"]
    assert validate_strategy_overrides_for_ids(
        ["dynamic_semantic"],
        {"dynamic_semantic": {"threshod": 0.9}},
    ) == ["dynamic_semantic received unsupported override keys: threshod"]


def test_openrouter_llm_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
    config = llm_config_from_env()
    assert config.provider == "openrouter"
    assert config.model == "anthropic/claude-3.5-sonnet"
    assert config.api_key_env == "OPENROUTER_API_KEY"
    assert config.base_url == "https://openrouter.ai/api/v1"


def test_provider_config_rejects_unknown_ids():
    try:
        embedding_config_from_env(provider="unknown", model="x")
    except ValueError as exc:
        assert "Unsupported embedding provider" in str(exc)
    else:
        raise AssertionError("unknown embedding provider should fail")

    try:
        llm_config_from_env(provider="unknown", model="x")
    except ValueError as exc:
        assert "Unsupported LLM provider" in str(exc)
    else:
        raise AssertionError("unknown LLM provider should fail")


def test_custom_embedding_config_requires_base_url(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    try:
        embedding_config_from_env(provider="custom", model="embedding-model")
    except ValueError as exc:
        assert "EMBEDDING_BASE_URL is required" in str(exc)
    else:
        raise AssertionError("custom embedding provider should require base_url")

    config = embedding_config_from_env(
        provider="custom",
        model="embedding-model",
        api_key_env="CUSTOM_EMBEDDING_KEY",
        base_url="https://example.test/v1",
    )
    assert config.provider == "custom"
    assert config.base_url == "https://example.test/v1"
    assert config.api_key_env == "CUSTOM_EMBEDDING_KEY"


def test_mock_embedding_provider_builds_without_network(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    config = embedding_config_from_env(provider="mock", model="mock:16")
    model = build_embedding_model(config)
    embedding = model.get_text_embedding("smoke")
    assert config.provider == "mock"
    assert len(embedding) == 16


def test_hpo_config_loads_default_search_space_and_objective():
    settings = load_hpo_settings(soft_token_limit=1200)
    assert "threshold" in settings.search_space
    assert settings.search_space["min_window"].type == "int"
    assert settings.objective.hr_weight == 100.0
    assert settings.objective.soft_token_limit == 1200


def test_shared_sentence_splitter_handles_common_abbreviations():
    text = "This e.g. example stays together. Fig. 3 shows the result. The value is 3.14."

    sentences = split_into_sentences(text)

    assert sentences == [
        "This e.g. example stays together.",
        "Fig. 3 shows the result.",
        "The value is 3.14.",
    ]


def test_answer_matching_contract_between_metrics_and_corpus_lookup():
    sentences = [
        "A distractor sentence about alpha beta.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    paraphrased = "quick brown fox jumped over lazy dog"

    # Sentence-level ground-truth resolution stays tolerant to paraphrase...
    assert find_answer_sentence_idx(sentences, paraphrased) == 1
    # ...while chunk-level hit metrics require (near-)verbatim containment.
    assert hit_rate([sentences[1]], paraphrased) == 0.0
    assert hit_rate([sentences[1]], sentences[1]) == 1.0


def test_optuna_sqlite_storage_parent_is_created(tmp_path):
    storage_path = tmp_path / "nested" / "study.db"

    ensure_sqlite_storage_parent(f"sqlite:///{storage_path}")

    assert storage_path.parent.exists()


def test_optuna_cli_reports_config_errors_without_traceback():
    bad_strategy = subprocess.run(
        [
            sys.executable,
            "run_optuna.py",
            "--strategy",
            "token_text",
            "--n-trials",
            "1",
            "--corpus-path",
            "data/does_not_matter.pkl",
        ],
        capture_output=True,
        text=True,
    )
    missing_corpus = subprocess.run(
        [
            sys.executable,
            "run_optuna.py",
            "--n-trials",
            "1",
            "--corpus-path",
            "data/does_not_exist.pkl",
        ],
        capture_output=True,
        text=True,
    )

    assert bad_strategy.returncode == 2
    assert "[ERROR] Cached Optuna currently supports only dynamic_semantic" in bad_strategy.stderr
    assert "Traceback" not in bad_strategy.stderr
    assert missing_corpus.returncode == 2
    assert "[ERROR] Corpus not found: data" in missing_corpus.stderr
    assert "Traceback" not in missing_corpus.stderr


def test_load_combined_benchmark_jsonl(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    record = {
        "title": "Doc",
        "text": "Alpha beta. Gamma delta.",
        "qa_pairs": [{"question": "What starts the doc?", "answer_sentence": "Alpha beta."}],
    }
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    items = load_benchmark_dataset(dataset_path=str(dataset_path))

    assert items[0]["title"] == "Doc"
    assert items[0]["qa_pairs"][0]["question"] == "What starts the doc?"


def test_load_combined_benchmark_jsonl_with_utf8_bom(tmp_path):
    dataset_path = tmp_path / "dataset_bom.jsonl"
    record = {
        "title": "Doc",
        "text": "Alpha beta. Gamma delta.",
        "qa_pairs": [{"question": "What starts the doc?", "answer_sentence": "Alpha beta."}],
    }
    dataset_path.write_text("﻿" + json.dumps(record) + "\n", encoding="utf-8")

    items = load_benchmark_dataset(dataset_path=str(dataset_path))

    assert items[0]["title"] == "Doc"


def test_load_paired_articles_questions_jsonl(tmp_path):
    articles_path = tmp_path / "articles.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    articles_path.write_text(
        json.dumps({"id": 10, "title": "Doc", "text": "Alpha beta. Gamma delta."}) + "\n",
        encoding="utf-8",
    )
    questions_path.write_text(
        json.dumps({
            "article_id": "10",
            "question": "What starts the doc?",
            "answer_sentence": "Alpha beta.",
        }) + "\n",
        encoding="utf-8",
    )

    items = load_benchmark_dataset(
        articles_path=str(articles_path),
        questions_path=str(questions_path),
    )

    assert items[0]["id"] == 10
    assert items[0]["qa_pairs"][0]["answer_sentence"] == "Alpha beta."


def test_summarize_new_benchmark_result_schema(tmp_path):
    result_path = tmp_path / "benchmark_new.json"
    result_path.write_text(
        json.dumps({
            "config": {
                "source": "custom",
                "dataset_name": "custom_sample",
                "top_k": 5,
                "strategies": "naive",
                "num_articles": 5,
                "num_questions": 3,
                "actual_num_articles": 2,
                "actual_num_questions": 6,
                "requested_num_articles": 5,
                "questions_per_article": 3,
            },
            "embedding": {"provider": "huggingface", "model": "test-model"},
            "llm": {
                "provider": "openrouter",
                "model": "openai/gpt-4.1-mini",
                "used_for_qa_generation": True,
            },
            "effective_strategy_overrides": {
                "Naive Chunking": {},
                "Dynamic Semantic": {"threshold": 0.9},
            },
            "aggregate": {
                "Naive Chunking": [
                    {"tokens": 10, "hr@5": 1.0, "mrr": 1.0},
                    {"tokens": 20, "hr@5": 0.0, "mrr": 0.0},
                ],
                "Dynamic Semantic": [
                    {"tokens": 12, "hr@5": 1.0, "mrr": 1.0},
                ],
            },
        }),
        encoding="utf-8",
    )

    rows = summarize_benchmark_file(result_path)
    by_strategy = {row["strategy"]: row for row in rows}

    assert by_strategy["Naive Chunking"]["dataset_name"] == "custom_sample"
    assert by_strategy["Naive Chunking"]["avg_tokens"] == 15
    assert by_strategy["Naive Chunking"]["avg_hr@5"] == 0.5
    assert by_strategy["Naive Chunking"]["metric_k"] == 5
    assert by_strategy["Naive Chunking"]["embedding_model"] == "test-model"
    assert by_strategy["Naive Chunking"]["llm_provider"] == "openrouter"
    assert by_strategy["Naive Chunking"]["llm_model"] == "openai/gpt-4.1-mini"
    assert by_strategy["Naive Chunking"]["llm_used_for_qa_generation"] is True
    assert by_strategy["Naive Chunking"]["strategy_overrides"] == "{}"
    assert by_strategy["Dynamic Semantic"]["strategy_overrides"] == '{"threshold": 0.9}'


def test_summarize_results_writes_only_requested_outputs(tmp_path):
    result_path = tmp_path / "benchmark_new.json"
    csv_path = tmp_path / "summary.csv"
    result_path.write_text(
        json.dumps({
            "config": {"source": "static", "dataset_name": "static"},
            "embedding": {"provider": "mock", "model": "mock:8"},
            "aggregate": {"Naive Chunking": [{"tokens": 10, "mrr": 1.0}]},
        }),
        encoding="utf-8",
    )

    process = subprocess.run(
        [sys.executable, "summarize_results.py", str(result_path), "--csv", str(csv_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert csv_path.exists()
    assert "CSV:" in process.stdout
    assert "JSONL:" not in process.stdout
    assert "Leaderboard CSV:" not in process.stdout


def test_build_leaderboard_ranks_quality_then_tokens():
    rows = [
        {"strategy": "B", "metric_k": 3, "avg_ndcg@3": 0.9, "avg_tokens": 200},
        {"strategy": "A", "metric_k": 3, "avg_ndcg@3": 0.9, "avg_tokens": 100},
        {"strategy": "C", "metric_k": 3, "avg_ndcg@3": 0.8, "avg_tokens": 50},
    ]

    leaderboard = build_leaderboard(rows)

    assert [row["strategy"] for row in leaderboard] == ["A", "B", "C"]
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[0]["rank_metric"] == "avg_ndcg@3"
    assert leaderboard[0]["rank_score"] == 0.9


def test_build_grouped_leaderboard_ranks_within_dataset():
    rows = [
        {"dataset_name": "b", "strategy": "B", "metric_k": 3, "avg_ndcg@3": 0.8, "avg_tokens": 200},
        {"dataset_name": "a", "strategy": "A", "metric_k": 3, "avg_ndcg@3": 0.9, "avg_tokens": 100},
        {"dataset_name": "b", "strategy": "A", "metric_k": 3, "avg_ndcg@3": 0.7, "avg_tokens": 50},
        {"dataset_name": "a", "strategy": "B", "metric_k": 3, "avg_ndcg@3": 0.8, "avg_tokens": 50},
    ]

    leaderboard = build_leaderboard(rows, group_by=["dataset_name"])

    assert [(row["dataset_name"], row["strategy"], row["rank"]) for row in leaderboard] == [
        ("a", "A", 1),
        ("a", "B", 2),
        ("b", "B", 1),
        ("b", "A", 2),
    ]
    assert leaderboard[0]["rank_group"] == "dataset_name=a"


def test_summarize_legacy_benchmark_result_schema(tmp_path):
    result_path = tmp_path / "benchmark_old.json"
    result_path.write_text(
        json.dumps({
            "config": {"source": "static"},
            "aggregate_metrics": {
                "Dynamic Semantic": {
                    "avg_tokens": 42,
                    "avg_hr@5": 1.0,
                }
            },
        }),
        encoding="utf-8",
    )

    rows = summarize_benchmark_file(result_path)

    assert rows[0]["strategy"] == "Dynamic Semantic"
    assert rows[0]["avg_tokens"] == 42
    assert rows[0]["avg_hr@5"] == 1.0


def test_dynamic_strategy_embeds_sentences_in_one_batch():
    from typing import ClassVar

    from llama_index.core import Document, Settings
    from llama_index.core.embeddings import MockEmbedding

    from src.strategies import DynamicSemanticStrategy

    class CountingEmbedding(MockEmbedding):
        batch_sizes: ClassVar[list[int]] = []

        def get_text_embedding_batch(self, texts, **kwargs):
            type(self).batch_sizes.append(len(texts))
            return super().get_text_embedding_batch(texts, **kwargs)

    CountingEmbedding.batch_sizes.clear()
    old_embed_model = getattr(Settings, "_embed_model", None)
    Settings.embed_model = CountingEmbedding(embed_dim=8)
    try:
        text = " ".join(
            f"Sentence number {i} talks about one shared topic." for i in range(12)
        )
        DynamicSemanticStrategy([Document(text=text)], top_k=3)
    finally:
        Settings._embed_model = old_embed_model

    # All 12 sentences must arrive in a single batched call, not 12 single calls
    assert CountingEmbedding.batch_sizes
    assert CountingEmbedding.batch_sizes[0] == 12
