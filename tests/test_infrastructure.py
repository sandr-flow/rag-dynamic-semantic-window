import json
import subprocess
import sys
from pathlib import Path

from quick_benchmark import QuickBenchmarkSelection, build_benchmark_command, discover_datasets
from run_experiments import _config_sha256, _expected_artifacts
from run_optuna import ensure_sqlite_storage_parent
from run_smoke import build_smoke_commands
from src.benchmark_datasets import load_benchmark_dataset
from src.experiment_config import build_commands, load_experiment_config, validate_experiment_config
from src.hpo_config import load_hpo_settings
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


def test_parse_strategy_aliases():
    assert parse_strategy_ids("sentence,dynamic") == ["naive", "dynamic_semantic"]
    assert "token_text" in parse_strategy_ids("default")
    assert "code" not in parse_strategy_ids("default")
    assert {"markdown", "html", "json", "code"}.issubset(set(parse_strategy_ids("all")))
    assert parse_strategy_ids("markdown-node-parser,code-splitter") == ["markdown", "code"]


def test_strategy_and_provider_catalogs_include_core_entries():
    strategies = {item["id"]: item for item in strategy_catalog()}
    providers = provider_catalog()

    assert strategies["dynamic_semantic"]["default"] is True
    assert "threshold" in strategies["dynamic_semantic"]["override_keys"]
    assert strategies["dynamic_semantic"]["default_params"]["threshold"] == default_strategy_params(
        "dynamic"
    )["threshold"]
    assert "chunk_size" in strategies["token_text"]["default_params"]
    assert isinstance(strategies["html"]["default_params"]["tags"], list)
    assert any(item["id"] == "openrouter" for item in providers["llms"])
    assert any(item["id"] == "mock" for item in providers["embeddings"])
    assert any(item["id"] == "custom" for item in providers["embeddings"])


def test_cli_inspection_commands_do_not_load_models():
    strategies = subprocess.run(
        [sys.executable, "run_benchmark.py", "--list-strategies"],
        check=True,
        capture_output=True,
        text=True,
    )
    providers = subprocess.run(
        [sys.executable, "run_benchmark.py", "--list-providers"],
        check=True,
        capture_output=True,
        text=True,
    )
    validate = subprocess.run(
        [sys.executable, "run_experiments.py", "configs/static_smoke.yaml", "--validate-only"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dynamic_semantic" in strategies.stdout
    assert "defaults:" in strategies.stdout
    assert "\"threshold\"" in strategies.stdout
    assert "openrouter" in providers.stdout
    assert "Experiment config OK: static_smoke" in validate.stdout


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


def test_smoke_cli_builds_offline_commands():
    quick_commands = build_smoke_commands()
    full_quality_commands = build_smoke_commands(
        full=True,
        quality=True,
        output_dir="results/smoke/test",
    )

    quick_names = [command.name for command in quick_commands]
    full_quality_names = [command.name for command in full_quality_commands]

    assert quick_names == [
        "list_strategies",
        "list_providers",
        "provider_dry_run",
        "mock_embedding_live",
        "quick_benchmark_print",
        "static_validate",
        "custom_validate",
        "custom_optuna_validate",
        "multi_dataset_optuna_validate",
    ]
    assert "custom_optuna_experiment" in full_quality_names
    assert "multi_dataset_optuna_experiment" in full_quality_names
    assert "pytest" in full_quality_names
    assert "pip_check" in full_quality_names
    assert all("wikipedia" not in " ".join(command.argv) for command in full_quality_commands)
    assert all("qasper" not in " ".join(command.argv) for command in full_quality_commands)


def test_smoke_cli_dry_run():
    process = subprocess.run(
        [sys.executable, "run_smoke.py", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Smoke commands: 9" in process.stdout
    assert "quick_benchmark_print" in process.stdout
    assert "custom_optuna_validate" in process.stdout
    assert "provider_dry_run" in process.stdout


def test_quick_benchmark_discovers_datasets_and_builds_command(tmp_path):
    dataset_path = tmp_path / "my_dataset.jsonl"
    articles_path = tmp_path / "articles.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    dataset_path.write_text("{}", encoding="utf-8")
    articles_path.write_text("{}", encoding="utf-8")
    questions_path.write_text("{}", encoding="utf-8")

    assert discover_datasets(tmp_path) == [dataset_path]

    command = build_benchmark_command(
        QuickBenchmarkSelection(
            source="custom",
            dataset_path="data/my_dataset.jsonl",
            dataset_name="my_dataset",
            embedding_provider="huggingface",
            embedding_model="BAAI/bge-m3",
            strategies="default",
            top_k=5,
            metric_k=5,
        )
    )

    assert command[1] == "run_benchmark.py"
    assert "--source" in command
    assert command[command.index("--source") + 1] == "custom"
    assert command[command.index("--dataset-path") + 1] == "data/my_dataset.jsonl"
    assert command[command.index("--embedding-provider") + 1] == "huggingface"
    assert command[command.index("--embedding-model") + 1] == "BAAI/bge-m3"
    assert "dynamic_semantic" in command[command.index("--strategies") + 1]


def test_quick_benchmark_print_command_with_args():
    process = subprocess.run(
        [
            sys.executable,
            "quick_benchmark.py",
            "data/custom_benchmark.jsonl",
            "--provider",
            "mock",
            "--model",
            "mock:64",
            "--strategies",
            "dynamic_semantic,token_text",
            "--print-command",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "run_benchmark.py" in process.stdout
    assert "--dataset-path data/custom_benchmark.jsonl" in process.stdout
    assert "--embedding-provider mock" in process.stdout
    assert "--embedding-model mock:64" in process.stdout
    assert "dynamic_semantic,token_text" in process.stdout


def test_benchmark_cli_reports_provider_errors_without_traceback():
    process = subprocess.run(
        [
            sys.executable,
            "run_benchmark.py",
            "--source",
            "static",
            "--embedding-provider",
            "typo",
            "--embedding-model",
            "x",
        ],
        capture_output=True,
        text=True,
    )

    assert process.returncode == 2
    assert "[ERROR] Unsupported embedding provider: typo" in process.stderr
    assert "Traceback" not in process.stderr


def test_code_splitter_dependency_available():
    from llama_index.core.node_parser import CodeSplitter

    splitter = CodeSplitter(language="python")
    assert splitter.language == "python"


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


def test_static_smoke_config_expands_to_benchmark_command():
    config = load_experiment_config(Path("configs") / "static_smoke.yaml")
    commands = build_commands(config)
    assert len(commands) == 1
    command = commands[0]
    assert command.name == "benchmark_e1_s1"
    assert "run_benchmark.py" in command.argv
    assert "--source" in command.argv
    assert "static" in command.argv
    assert "--strategies" in command.argv
    assert "--metric-k" in command.argv
    assert any("dynamic_semantic" in arg for arg in command.argv)


def test_custom_smoke_config_uses_checked_in_dataset():
    config = load_experiment_config(Path("configs") / "custom_smoke.yaml")
    commands = build_commands(config)
    command = commands[0]

    assert len(commands) == 1
    assert Path("data/custom_benchmark.jsonl").exists()
    assert "--dataset-path" in command.argv
    assert "data/custom_benchmark.jsonl" in command.argv
    assert "--embedding-provider" in command.argv
    assert "mock" in command.argv


def test_hpo_config_loads_search_space_and_objective():
    settings = load_hpo_settings("configs/hpo_dynamic_balanced.yaml")
    assert settings.search_space["threshold"].low == 0.5
    assert settings.search_space["threshold"].step == 0.001
    assert settings.search_space["min_window"].type == "int"
    assert settings.objective.hr_weight == 100.0
    assert settings.objective.soft_token_limit == 1200


def test_optuna_sqlite_storage_parent_is_created(tmp_path):
    storage_path = tmp_path / "nested" / "study.db"

    ensure_sqlite_storage_parent(f"sqlite:///{storage_path}")

    assert storage_path.parent.exists()


def test_custom_optuna_config_expands_hpo_config_flag():
    config = load_experiment_config(Path("configs") / "custom_optuna_example.yaml")
    commands = build_commands(config)
    prepare_command = next(command for command in commands if command.name == "prepare_corpus")
    optuna_command = next(command for command in commands if command.name == "optuna")
    benchmark_command = next(command for command in commands if command.name.startswith("benchmark"))

    assert "--corpus-path" in prepare_command.argv
    assert "--min-sentences" in prepare_command.argv
    assert "1" in prepare_command.argv
    assert "results/corpora/custom_example/cached_corpus.pkl" in prepare_command.argv
    assert "--articles-output-path" in prepare_command.argv
    assert "--questions-output-path" in prepare_command.argv
    assert "results/corpora/custom_example/cached_corpus.pkl" in optuna_command.argv
    assert "--hpo-config" in optuna_command.argv
    assert "configs/hpo_dynamic_balanced.yaml" in optuna_command.argv
    assert "--output-dir" in optuna_command.argv
    assert "results/optuna/custom_example" in optuna_command.argv
    assert "--best-params-json" in optuna_command.argv
    assert "results/optuna/custom_example/best_params.json" in optuna_command.argv
    assert "--strategy-overrides-json" in benchmark_command.argv
    assert "results/optuna/custom_example/best_params.json" in benchmark_command.argv


def test_custom_optuna_smoke_config_is_offline_full_pipeline():
    config = load_experiment_config(Path("configs") / "custom_optuna_smoke.yaml")
    commands = build_commands(config)

    assert validate_experiment_config(config) == []
    assert [command.name for command in commands] == [
        "prepare_corpus",
        "optuna",
        "benchmark_e1_s1",
    ]
    assert "mock" in commands[0].argv
    assert "--n-trials" in commands[1].argv
    assert "2" in commands[1].argv
    assert "results/optuna/custom_optuna_smoke/best_params.json" in commands[1].argv
    assert "results/optuna/custom_optuna_smoke/best_params.json" in commands[2].argv


def test_experiment_manifest_records_expected_artifacts(tmp_path):
    config = load_experiment_config(Path("configs") / "custom_optuna_smoke.yaml")
    commands = build_commands(config)

    assert _config_sha256(config) == _config_sha256(load_experiment_config(Path("configs") / "custom_optuna_smoke.yaml"))
    assert _expected_artifacts(commands[0].argv) == {
        "corpus": ["results/corpora/custom_optuna_smoke/cached_corpus.pkl"],
        "articles": ["results/corpora/custom_optuna_smoke/articles.jsonl"],
        "questions": ["results/corpora/custom_optuna_smoke/questions.jsonl"],
    }
    assert _expected_artifacts(commands[1].argv) == {
        "best_params_json": ["results/optuna/custom_optuna_smoke/best_params.json"],
        "best_params_txt": ["results/optuna/custom_optuna_smoke/best_params.txt"],
        "output_dir": ["results/optuna/custom_optuna_smoke"],
        "storage": ["results/optuna/custom_optuna_smoke/study.db"],
    }
    assert _expected_artifacts(commands[2].argv) == {
        "benchmark_results": ["results/benchmark_*.json"],
    }

    output_dir = tmp_path / "experiment"
    process = subprocess.run(
        [
            sys.executable,
            "run_experiments.py",
            "configs/custom_optuna_smoke.yaml",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "Manifest:" in process.stdout
    assert manifest["dry_run"] is True
    assert manifest["command_count"] == 3
    assert manifest["config_sha256"] == _config_sha256(config)
    assert manifest["runner"]["python"] == sys.executable
    assert manifest["runs"][0]["expected_artifacts"]["corpus"] == [
        "results/corpora/custom_optuna_smoke/cached_corpus.pkl"
    ]
    assert manifest["runs"][1]["expected_artifacts"]["best_params_json"] == [
        "results/optuna/custom_optuna_smoke/best_params.json"
    ]


def test_multi_dataset_optuna_smoke_expands_independent_pipelines():
    config = load_experiment_config(Path("configs") / "multi_dataset_optuna_smoke.yaml")
    commands = build_commands(config)

    assert validate_experiment_config(config) == []
    assert [command.name for command in commands] == [
        "custom_sample_prepare_corpus",
        "custom_sample_optuna",
        "custom_sample_benchmark_e1_s1",
        "optimization_sample_prepare_corpus",
        "optimization_sample_optuna",
        "optimization_sample_benchmark_e1_s1",
    ]
    assert "results/corpora/multi_dataset_optuna/custom_sample/cached_corpus.pkl" in commands[0].argv
    assert "results/optuna/multi_dataset_optuna/custom_sample/best_params.json" in commands[1].argv
    assert "results/optuna/multi_dataset_optuna/custom_sample/best_params.json" in commands[2].argv
    assert "custom_sample" in commands[2].argv
    assert "data/custom_benchmark_alt.jsonl" in commands[3].argv
    assert "optimization_sample" in commands[5].argv


def test_experiment_validation_rejects_prepare_optuna_corpus_mismatch():
    config = {
        "prepare_corpus": {
            "enabled": True,
            "source": "custom",
            "dataset_path": "data/custom_benchmark.jsonl",
            "corpus_path": "results/corpora/a.pkl",
        },
        "optuna": {
            "enabled": True,
            "corpus_path": "results/corpora/b.pkl",
        },
        "benchmark": {"enabled": False},
    }

    issues = validate_experiment_config(config)

    assert len(issues) == 1
    assert "prepare_corpus.corpus_path must match optuna.corpus_path" in issues[0].message


def test_experiment_validation_rejects_bad_optuna_config():
    config = {
        "benchmark": {"enabled": False},
        "optuna": {
            "enabled": True,
            "strategy": "token_text",
            "n_trials": 0,
        },
    }

    issues = validate_experiment_config(config)
    messages = [issue.message for issue in issues]

    assert any("optuna.strategy is unsupported" in message for message in messages)
    assert any("optuna.n_trials must be >= 1" in message for message in messages)


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


def test_provider_matrix_config_expands_embedding_llm_strategy_axes():
    config = load_experiment_config(Path("configs") / "provider_matrix_example.yaml")
    commands = build_commands(config)

    assert validate_experiment_config(config) == []
    assert len(commands) == 12
    assert commands[0].name == "benchmark_e1_l1_s1"
    assert "--llm-provider" in commands[0].argv
    assert "openrouter" in commands[0].argv
    assert "--embedding-provider" in commands[0].argv
    assert "huggingface" in commands[0].argv
    assert "--metric-k" in commands[0].argv
    assert "--qa-delay" in commands[0].argv


def test_dataset_matrix_config_expands_dataset_axis():
    config = load_experiment_config(Path("configs") / "dataset_matrix_smoke.yaml")
    commands = build_commands(config)

    assert validate_experiment_config(config) == []
    assert len(commands) == 2
    assert [command.name for command in commands] == ["benchmark_d1_e1_s1", "benchmark_d2_e1_s1"]
    assert commands[0].argv[commands[0].argv.index("--source") + 1] == "static"
    assert commands[0].argv[commands[0].argv.index("--dataset-name") + 1] == "static"
    assert "--dataset-path" not in commands[0].argv
    assert commands[1].argv[commands[1].argv.index("--source") + 1] == "custom"
    assert commands[1].argv[commands[1].argv.index("--dataset-name") + 1] == "custom_sample"
    assert "data/custom_benchmark.jsonl" in commands[1].argv


def test_experiment_validation_rejects_missing_custom_dataset():
    config = {
        "benchmark": {
            "enabled": True,
            "source": "custom",
            "dataset_path": "data/does_not_exist.jsonl",
        },
        "embeddings": [{"provider": "mock", "model": "mock:16"}],
    }

    issues = validate_experiment_config(config)

    assert len(issues) == 1
    assert issues[0].level == "error"
    assert "does_not_exist.jsonl" in issues[0].message


def test_experiment_validation_rejects_bad_provider_config(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    config = {
        "benchmark": {"enabled": True, "source": "static"},
        "embeddings": [
            {"provider": "hugingface", "model": "x"},
            {"provider": "custom", "model": "embedding-model"},
        ],
        "llms": [{"provider": "custom", "model": ""}],
    }

    issues = validate_experiment_config(config)
    messages = [issue.message for issue in issues]

    assert any("embeddings[1].provider is unsupported" in message for message in messages)
    assert any("embeddings[2].base_url is required" in message for message in messages)
    assert any("llms[1].model must not be empty" in message for message in messages)
    assert any("llms[1].base_url is required" in message for message in messages)


def test_experiment_validation_rejects_empty_axes_and_unknown_strategies():
    empty_axis_config = {
        "benchmark": {"enabled": True, "source": "static"},
        "datasets": [],
        "embeddings": [],
        "llms": [],
        "strategy_sets": [],
    }

    issues = validate_experiment_config(empty_axis_config)
    messages = [issue.message for issue in issues]

    assert "datasets must not be empty" in messages
    assert "embeddings must not be empty" in messages
    assert "llms must not be empty" in messages
    assert "strategy_sets must not be empty" in messages

    try:
        build_commands(empty_axis_config)
    except ValueError as exc:
        assert "embeddings must not be empty" in str(exc)
    else:
        raise AssertionError("build_commands must reject explicitly empty benchmark axes")

    unknown_strategy_config = {
        "benchmark": {"enabled": True, "source": "static"},
        "strategy_sets": [{"strategies": ["dynamic_semantic", "missing_strategy"]}],
    }

    issues = validate_experiment_config(unknown_strategy_config)

    assert len(issues) == 1
    assert "missing_strategy" in issues[0].message


def test_experiment_validation_checks_strategy_override_keys(tmp_path):
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps({"dynamic_semantic": {"threshod": 0.9}}), encoding="utf-8")
    config = {
        "strategy_overrides_json": str(overrides_path),
        "benchmark": {"enabled": True, "source": "static"},
        "embeddings": [{"provider": "mock", "model": "mock:16"}],
        "strategy_sets": [{"strategies": ["dynamic_semantic"]}],
    }

    issues = validate_experiment_config(config)

    assert len(issues) == 1
    assert "threshod" in issues[0].message


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
    dataset_path.write_text("\ufeff" + json.dumps(record) + "\n", encoding="utf-8")

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
    assert by_strategy["Naive Chunking"]["num_articles"] == 2
    assert by_strategy["Naive Chunking"]["num_questions"] == 6
    assert by_strategy["Naive Chunking"]["requested_num_articles"] == 5
    assert by_strategy["Naive Chunking"]["questions_per_article"] == 3
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
