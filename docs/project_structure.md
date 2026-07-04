# Project Structure

This repository keeps CLI entrypoints in the project root for backwards-compatible usage,
while reusable implementation lives under `src/`.

## Core Entrypoints

- `run_benchmark.py` - compare chunking/retrieval strategies on static, built-in, or custom datasets.
- `quick_benchmark.py` - interactive wrapper for manual dataset/model/strategy selection.
- `check_providers.py` - validate embedding/LLM provider config and optional live connectivity.
- `prepare_corpus.py` - build cached corpora for fast Optuna trials.
- `run_optuna.py` - optimize Dynamic Semantic hyperparameters on a cached corpus.
- `run_experiments.py` - run YAML/JSON experiment matrices and write manifests/logs.
- `run_smoke.py` - run offline smoke checks for catalogs, providers, configs, and optional full pipelines.
- `summarize_results.py` - export benchmark result JSON files to CSV/JSONL summaries and leaderboard CSV.

Useful inspection commands:

- `python run_benchmark.py --list-strategies`
- `python run_benchmark.py --list-providers`
- `python run_experiments.py configs/static_smoke.yaml --validate-only`
- `python run_smoke.py --dry-run`

## Configuration

- `configs/static_smoke.yaml` - minimal benchmark matrix example.
- `configs/custom_smoke.yaml` - custom dataset smoke run with mock embeddings.
- `configs/custom_optuna_smoke.yaml` - full offline prepare + Optuna + benchmark smoke.
- `configs/custom_optuna_example.yaml` - custom dataset + Optuna + benchmark workflow.
- `configs/provider_matrix_example.yaml` - provider and strategy matrix example.
- `configs/hpo_dynamic_balanced.yaml` - configurable Dynamic Semantic HPO search space.
- `.env.example` - provider/API-key/environment template.
- `requirements.txt` - runtime and experiment dependencies.
- `requirements-dev.txt` - local test/lint tools layered on top of runtime deps.
- `docs/target_state.md` - acceptance checklist for the intended benchmark stand.
- `docs/extension_guide.md` - checklist for adding dataset, provider, strategy, and HPO adapters.

## Runtime Modules

- `src/strategies.py` - retrieval strategy implementations.
- `src/strategy_registry.py` - strategy ids, aliases, and factory.
- `src/expansion_core.py` - the single Dynamic Semantic expansion implementation
  (pure numpy core shared by the benchmark and HPO paths).
- `src/dynamic_retriever.py` - LlamaIndex adapter over the expansion core.
- `src/providers.py` - embedding and LLM provider factories, including OpenRouter.
- `src/benchmark_datasets.py` - custom benchmark dataset loaders.
- `src/experiment_config.py` - experiment YAML/JSON command expansion.
- `src/hpo_config.py` - Optuna search-space and objective-policy config.
- `src/result_summary.py` - benchmark result flattening and export helpers.

## Data and Outputs

- `data/` - input datasets and cached corpora.
- `results/` - benchmark, Optuna, experiment, and analysis outputs. Ignored by git.
  Generated corpora can live under `results/corpora/<dataset>/`.
  Prefer dataset-specific HPO paths such as `results/optuna/qasper/best_params.json`.
- `tests/` - fast infrastructure tests that avoid network and model downloads.
- `pyproject.toml` - pytest and Ruff defaults.

## Legacy/Analysis Scripts

The following root scripts are retained for compatibility with prior experiments:

- `run_comparison.py`
- `evaluate_params.py`
- `analyze_failures.py`
- `benchmark_failures.py`
- `prepare_failures_corpus.py`
