"""Experiment configuration loading and command expansion."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.providers import DEFAULT_CHAT_MODELS, DEFAULT_EMBEDDING_MODELS
from src.strategy_registry import parse_strategy_ids, validate_strategy_overrides_for_ids


@dataclass
class CommandSpec:
    """A command generated from an experiment config."""

    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """A config validation issue."""

    level: str
    message: str


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config from YAML or JSON."""
    config_path = Path(path)
    with open(config_path, encoding="utf-8-sig") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(f)
        else:
            payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a mapping")
    return payload


def validate_experiment_config(
    config: dict[str, Any],
    base_dir: str | Path = ".",
) -> list[ValidationIssue]:
    """Validate file references in an experiment config."""
    if config.get("pipelines"):
        return _validate_pipeline_configs(config, base_dir)

    return _validate_single_experiment_config(config, base_dir)


def _validate_single_experiment_config(
    config: dict[str, Any],
    base_dir: str | Path = ".",
) -> list[ValidationIssue]:
    base_path = Path(base_dir)
    issues: list[ValidationIssue] = []

    prepare = config.get("prepare_corpus") or {}
    benchmark = config.get("benchmark") or {}
    optuna = config.get("optuna") or {}
    benchmark_matrix_is_valid = True

    if prepare.get("enabled", False) and prepare.get("source") == "custom":
        _validate_custom_dataset_refs(prepare, base_path, issues, "prepare_corpus")

    if benchmark.get("enabled", True):
        benchmark_matrix_is_valid = _validate_benchmark_matrix(config, issues)

    if benchmark.get("enabled", True) and benchmark_matrix_is_valid:
        for index, benchmark_variant in enumerate(_benchmark_variants(benchmark, config.get("datasets")), start=1):
            label = "benchmark" if not config.get("datasets") else f"datasets[{index}]"
            if benchmark_variant.get("source") == "custom":
                _validate_custom_dataset_refs(benchmark_variant, base_path, issues, label)

    _validate_provider_sections(config, prepare, benchmark, issues)

    hpo_config = optuna.get("hpo_config")
    if optuna.get("enabled", False) and hpo_config:
        _check_file(hpo_config, base_path, issues, "optuna.hpo_config")
    if optuna.get("enabled", False):
        _validate_optuna_config(optuna, issues, "optuna")

    corpus_path = optuna.get("corpus_path")
    prepare_will_run = bool(prepare.get("enabled", False))
    if optuna.get("enabled", False) and corpus_path and not prepare_will_run:
        _check_file(corpus_path, base_path, issues, "optuna.corpus_path")
    if optuna.get("enabled", False) and corpus_path and prepare_will_run:
        prepare_corpus_path = prepare.get("corpus_path") or "data/cached_corpus.pkl"
        if str(prepare_corpus_path) != str(corpus_path):
            issues.append(
                ValidationIssue(
                    level="error",
                    message=(
                        "prepare_corpus.corpus_path must match optuna.corpus_path "
                        f"when both phases run: {prepare_corpus_path} != {corpus_path}"
                    ),
                )
            )

    overrides_path = config.get("strategy_overrides_json")
    optuna_will_write_best_params = bool(optuna.get("enabled", False) and optuna.get("apply_best_params", False))
    if overrides_path and not optuna_will_write_best_params:
        _check_file(overrides_path, base_path, issues, "strategy_overrides_json")
        resolved_overrides_path = _resolve_path(overrides_path, base_path)
        if resolved_overrides_path.exists() and benchmark_matrix_is_valid:
            _validate_strategy_overrides_file(resolved_overrides_path, config, issues)

    return issues


def _validate_optuna_config(
    optuna: dict[str, Any],
    issues: list[ValidationIssue],
    label: str,
) -> None:
    strategy = str(optuna.get("strategy") or "dynamic_semantic").replace("-", "_").lower()
    if strategy not in {"dynamic", "dynamic_semantic"}:
        issues.append(
            ValidationIssue(
                level="error",
                message=f"{label}.strategy is unsupported for cached HPO: {strategy}. Supported: dynamic_semantic",
            )
        )

    n_trials = optuna.get("n_trials")
    if n_trials is not None:
        try:
            parsed_n_trials = int(n_trials)
        except (TypeError, ValueError):
            issues.append(ValidationIssue(level="error", message=f"{label}.n_trials must be an integer"))
            return
        if parsed_n_trials < 1:
            issues.append(ValidationIssue(level="error", message=f"{label}.n_trials must be >= 1"))


def build_commands(config: dict[str, Any]) -> list[CommandSpec]:
    """Build command specs for prepare/optuna/benchmark phases."""
    if config.get("pipelines"):
        return _build_pipeline_commands(config)

    return _build_single_experiment_commands(config)


def _build_single_experiment_commands(config: dict[str, Any]) -> list[CommandSpec]:
    commands: list[CommandSpec] = []

    prepare = config.get("prepare_corpus") or {}
    if prepare.get("enabled", False):
        commands.append(_build_prepare_command(prepare, config))

    optuna = config.get("optuna") or {}
    best_params_path = config.get("strategy_overrides_json")
    if optuna.get("enabled", False):
        commands.append(_build_optuna_command(optuna))
        if optuna.get("apply_best_params", False) and not best_params_path:
            best_params_path = _optuna_best_params_json_path(optuna)

    benchmark = config.get("benchmark") or {}
    if benchmark.get("enabled", True):
        commands.extend(_build_benchmark_commands(config, best_params_path))

    if not commands:
        raise ValueError("Experiment config did not enable any runnable phases")

    return commands


def _validate_pipeline_configs(
    config: dict[str, Any],
    base_dir: str | Path,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    base_config = _base_pipeline_config(config)
    for index, pipeline in enumerate(_as_list(config.get("pipelines")), start=1):
        if not isinstance(pipeline, dict):
            issues.append(ValidationIssue(level="error", message=f"pipelines[{index}] must be a mapping"))
            continue
        merged = _deep_merge(base_config, pipeline)
        for issue in _validate_single_experiment_config(merged, base_dir):
            issues.append(
                ValidationIssue(
                    level=issue.level,
                    message=f"pipelines[{index}]: {issue.message}",
                )
            )
    return issues


def _build_pipeline_commands(config: dict[str, Any]) -> list[CommandSpec]:
    commands: list[CommandSpec] = []
    base_config = _base_pipeline_config(config)
    for index, pipeline in enumerate(_as_list(config.get("pipelines")), start=1):
        if not isinstance(pipeline, dict):
            raise ValueError(f"pipelines[{index}] must be a mapping")
        merged = _deep_merge(base_config, pipeline)
        prefix = _command_name_part(pipeline.get("name") or f"p{index}")
        for command in _build_single_experiment_commands(merged):
            commands.append(
                CommandSpec(
                    name=f"{prefix}_{command.name}",
                    argv=command.argv,
                    env=command.env,
                )
            )

    if not commands:
        raise ValueError("Experiment config did not enable any runnable phases")
    return commands


def _base_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"pipelines", "name"}}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _command_name_part(value: Any) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    normalized = "_".join(part for part in "".join(chars).split("_") if part)
    return normalized or "pipeline"


def _build_prepare_command(prepare: dict[str, Any], config: dict[str, Any]) -> CommandSpec:
    argv = [sys.executable, "prepare_corpus.py"]
    _append_options(
        argv,
        prepare,
        {
            "source": "--source",
            "num_articles": "--num-articles",
            "min_length": "--min-length",
            "questions_per_article": "--questions-per-article",
            "top_k": "--top-k",
            "min_sentences": "--min-sentences",
            "corpus_path": "--corpus-path",
            "articles_output_path": "--articles-output-path",
            "questions_output_path": "--questions-output-path",
            "dataset_path": "--dataset-path",
            "articles_path": "--articles-path",
            "questions_path": "--questions-path",
            "qa_delay": "--qa-delay",
        },
    )
    _append_provider_options(argv, prepare.get("embedding") or config.get("embedding") or {}, "embedding")
    _append_provider_options(argv, prepare.get("llm") or config.get("llm") or {}, "llm")
    return CommandSpec(name="prepare_corpus", argv=argv, env=_env_overrides(config))


def _build_optuna_command(optuna: dict[str, Any]) -> CommandSpec:
    argv = [sys.executable, "run_optuna.py"]
    _append_options(
        argv,
        optuna,
        {
            "n_trials": "--n-trials",
            "study_name": "--study-name",
            "corpus_path": "--corpus-path",
            "strategy": "--strategy",
            "target_clusters": "--target-clusters",
            "soft_token_limit": "--soft-token-limit",
            "hpo_config": "--hpo-config",
            "storage": "--storage",
            "output_dir": "--output-dir",
            "best_params_json": "--best-params-json",
            "best_params_txt": "--best-params-txt",
        },
    )
    if optuna.get("no_viz", False):
        argv.append("--no-viz")
    return CommandSpec(name="optuna", argv=argv)


def _build_benchmark_commands(
    config: dict[str, Any],
    strategy_overrides_json: str | None,
) -> list[CommandSpec]:
    benchmark = config.get("benchmark") or {}
    embeddings = _benchmark_axis(config, plural_key="embeddings", singular_key="embedding", default={})
    llms = _benchmark_axis(config, plural_key="llms", singular_key="llm", default={})
    strategy_sets = _benchmark_axis(
        config,
        plural_key="strategy_sets",
        singular_key=None,
        default={"strategies": config.get("strategies", "default")},
    )
    benchmark_variants = _benchmark_variants(benchmark, config.get("datasets"))
    has_llm_axis = bool(config.get("llms"))
    has_dataset_axis = bool(config.get("datasets"))

    commands = []
    for dataset_idx, benchmark_variant in enumerate(benchmark_variants):
        for embedding_idx, embedding in enumerate(embeddings):
            for llm_idx, llm in enumerate(llms):
                for strategy_idx, strategy_set in enumerate(strategy_sets):
                    strategy_value = strategy_set.get("strategies", strategy_set) if isinstance(strategy_set, dict) else strategy_set
                    strategy_ids = parse_strategy_ids(_csv(strategy_value))

                    argv = [sys.executable, "run_benchmark.py"]
                    _append_options(
                        argv,
                        benchmark_variant,
                        {
                            "source": "--source",
                            "num_questions": "--num-questions",
                            "num_articles": "--num-articles",
                            "article": "--article",
                            "min_length": "--min-length",
                            "skip": "--skip",
                            "split": "--split",
                            "top_k": "--top-k",
                            "metric_k": "--metric-k",
                            "dataset_path": "--dataset-path",
                            "name": "--dataset-name",
                            "dataset_name": "--dataset-name",
                            "articles_path": "--articles-path",
                            "questions_path": "--questions-path",
                            "qa_delay": "--qa-delay",
                        },
                    )
                    argv.extend(["--strategies", ",".join(strategy_ids)])
                    if strategy_overrides_json:
                        argv.extend(["--strategy-overrides-json", strategy_overrides_json])
                    _append_provider_options(argv, embedding or {}, "embedding")
                    _append_provider_options(argv, llm or {}, "llm")

                    name_parts = ["benchmark"]
                    if has_dataset_axis:
                        name_parts.append(f"d{dataset_idx + 1}")
                    name_parts.append(f"e{embedding_idx + 1}")
                    if has_llm_axis:
                        name_parts.append(f"l{llm_idx + 1}")
                    name_parts.append(f"s{strategy_idx + 1}")
                    commands.append(CommandSpec(name="_".join(name_parts), argv=argv, env=_env_overrides(config)))

    return commands


def _benchmark_variants(
    benchmark: dict[str, Any],
    datasets: Any = None,
) -> list[dict[str, Any]]:
    if isinstance(datasets, list) and not datasets:
        raise ValueError("datasets must not be empty")
    if not datasets:
        return [benchmark]
    dataset_items = _as_list(datasets)
    variants = []
    for dataset in dataset_items:
        if not isinstance(dataset, dict):
            raise ValueError("datasets entries must be mappings")
        variant = dict(benchmark)
        variant.update(dataset)
        variants.append(variant)
    return variants


def _benchmark_axis(
    config: dict[str, Any],
    plural_key: str,
    singular_key: str | None,
    default: Any,
) -> list[Any]:
    if plural_key in config:
        value = config[plural_key]
        if isinstance(value, list) and not value:
            raise ValueError(f"{plural_key} must not be empty")
        return _as_list(value)
    if singular_key and singular_key in config:
        return _as_list(config.get(singular_key) or {})
    return _as_list(default)


def _append_provider_options(argv: list[str], provider_config: dict[str, Any], prefix: str) -> None:
    mapping = {
        "provider": f"--{prefix}-provider",
        "model": f"--{prefix}-model",
        "api_key_env": f"--{prefix}-api-key-env",
        "base_url": f"--{prefix}-base-url",
    }
    _append_options(argv, provider_config, mapping)


def _append_options(argv: list[str], source: dict[str, Any], mapping: dict[str, str]) -> None:
    for key, flag in mapping.items():
        value = source.get(key)
        if value is None:
            continue
        argv.extend([flag, str(value)])


def _env_overrides(config: dict[str, Any]) -> dict[str, str]:
    env = config.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("env must be a mapping")
    return {str(key): str(value) for key, value in env.items()}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _csv(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _validate_custom_dataset_refs(
    section: dict[str, Any],
    base_path: Path,
    issues: list[ValidationIssue],
    label: str,
) -> None:
    dataset_path = section.get("dataset_path")
    articles_path = section.get("articles_path")
    questions_path = section.get("questions_path")

    if dataset_path:
        _check_file(dataset_path, base_path, issues, f"{label}.dataset_path")
        return

    if articles_path and questions_path:
        _check_file(articles_path, base_path, issues, f"{label}.articles_path")
        _check_file(questions_path, base_path, issues, f"{label}.questions_path")
        return

    issues.append(
        ValidationIssue(
            level="error",
            message=f"{label} source=custom requires dataset_path or both articles_path and questions_path",
        )
    )


def _validate_benchmark_matrix(config: dict[str, Any], issues: list[ValidationIssue]) -> bool:
    ok = True
    for key in ("datasets", "embeddings", "llms", "strategy_sets"):
        value = config.get(key)
        if isinstance(value, list) and not value:
            issues.append(ValidationIssue(level="error", message=f"{key} must not be empty"))
            ok = False

    datasets = config.get("datasets")
    if datasets is not None:
        for index, dataset in enumerate(_as_list(datasets), start=1):
            if not isinstance(dataset, dict):
                issues.append(
                    ValidationIssue(
                        level="error",
                        message=f"datasets[{index}] must be a mapping",
                    )
                )
                ok = False

    if "strategies" in config and isinstance(config["strategies"], list) and not config["strategies"]:
        issues.append(ValidationIssue(level="error", message="strategies must not be empty"))
        ok = False

    strategy_sets = config.get("strategy_sets")
    if not (isinstance(strategy_sets, list) and not strategy_sets):
        for index, strategy_set in enumerate(_configured_strategy_values(config), start=1):
            try:
                parse_strategy_ids(_csv(strategy_set))
            except ValueError as exc:
                issues.append(
                    ValidationIssue(
                        level="error",
                        message=f"strategy_sets[{index}].strategies is invalid: {exc}",
                    )
                )
                ok = False

    return ok


def _validate_provider_sections(
    config: dict[str, Any],
    prepare: dict[str, Any],
    benchmark: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    _validate_provider_config(prepare.get("embedding"), "embedding", "prepare_corpus.embedding", issues)
    _validate_provider_config(prepare.get("llm"), "llm", "prepare_corpus.llm", issues)
    _validate_provider_config(config.get("embedding"), "embedding", "embedding", issues)
    _validate_provider_config(config.get("llm"), "llm", "llm", issues)

    for index, provider_config in enumerate(_as_list(config.get("embeddings")), start=1):
        _validate_provider_config(provider_config, "embedding", f"embeddings[{index}]", issues)

    for index, provider_config in enumerate(_as_list(config.get("llms")), start=1):
        _validate_provider_config(provider_config, "llm", f"llms[{index}]", issues)

    benchmark_llm = benchmark.get("llm")
    if benchmark_llm is not None:
        _validate_provider_config(benchmark_llm, "llm", "benchmark.llm", issues)


def _validate_provider_config(
    provider_config: Any,
    kind: str,
    label: str,
    issues: list[ValidationIssue],
) -> None:
    if provider_config is None:
        return
    if not isinstance(provider_config, dict):
        issues.append(ValidationIssue(level="error", message=f"{label} must be a mapping"))
        return

    provider = provider_config.get("provider")
    if provider is None:
        return
    provider_id = str(provider).lower()

    supported = DEFAULT_EMBEDDING_MODELS if kind == "embedding" else DEFAULT_CHAT_MODELS
    if provider_id not in supported:
        choices = ", ".join(sorted(supported))
        issues.append(
            ValidationIssue(
                level="error",
                message=f"{label}.provider is unsupported: {provider_id}. Supported: {choices}",
            )
        )
        return

    model = provider_config.get("model")
    if model is not None and not str(model).strip():
        issues.append(ValidationIssue(level="error", message=f"{label}.model must not be empty"))

    if kind == "embedding" and provider_id == "custom":
        base_url = provider_config.get("base_url") or os.getenv("EMBEDDING_BASE_URL")
        if not base_url:
            issues.append(
                ValidationIssue(
                    level="error",
                    message=f"{label}.base_url is required for custom embedding provider",
                )
            )

    if kind == "llm" and provider_id == "custom":
        base_url = provider_config.get("base_url") or os.getenv("LLM_BASE_URL")
        if not base_url:
            issues.append(
                ValidationIssue(
                    level="error",
                    message=f"{label}.base_url is required for custom LLM provider",
                )
            )


def _check_file(
    path: str,
    base_path: Path,
    issues: list[ValidationIssue],
    label: str,
) -> None:
    candidate = _resolve_path(path, base_path)
    if not candidate.exists():
        issues.append(
            ValidationIssue(
                level="error",
                message=f"{label} does not exist: {path}",
            )
        )


def _resolve_path(path: str, base_path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base_path / candidate
    return candidate


def _validate_strategy_overrides_file(
    overrides_path: Path,
    config: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    try:
        with open(overrides_path, encoding="utf-8-sig") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        issues.append(
            ValidationIssue(
                level="error",
                message=f"strategy_overrides_json is not valid JSON: {exc}",
            )
        )
        return

    overrides = payload.get("params") if isinstance(payload, dict) and isinstance(payload.get("params"), dict) else payload
    if not isinstance(overrides, dict):
        issues.append(
            ValidationIssue(
                level="error",
                message="strategy_overrides_json must contain a JSON object or Optuna best_params object",
            )
        )
        return

    for strategy_ids in _configured_strategy_id_sets(config):
        for error in validate_strategy_overrides_for_ids(strategy_ids, overrides):
            issues.append(ValidationIssue(level="error", message=f"strategy_overrides_json: {error}"))


def _configured_strategy_id_sets(config: dict[str, Any]) -> list[list[str]]:
    return [parse_strategy_ids(_csv(strategy_value)) for strategy_value in _configured_strategy_values(config)]


def _configured_strategy_values(config: dict[str, Any]) -> list[Any]:
    strategy_sets = _benchmark_axis(
        config,
        plural_key="strategy_sets",
        singular_key=None,
        default={"strategies": config.get("strategies", "default")},
    )
    values = []
    for strategy_set in strategy_sets:
        if isinstance(strategy_set, dict):
            values.append(strategy_set.get("strategies", strategy_set))
        else:
            values.append(strategy_set)
    return values


def _optuna_best_params_json_path(optuna: dict[str, Any]) -> str:
    if optuna.get("best_params_json"):
        return str(optuna["best_params_json"])
    if optuna.get("output_dir"):
        return str(Path(str(optuna["output_dir"])) / "best_params.json")
    return str(Path("results") / "best_params.json")
