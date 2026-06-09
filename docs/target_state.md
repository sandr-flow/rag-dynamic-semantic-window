# Target State

This document fixes the intended end state for the repository. It is the working
acceptance checklist for turning the project into a reusable chunking benchmark
stand without changing the Dynamic Semantic algorithm itself.

## Product Shape

The repository is a reproducible CLI-first benchmark stand for comparing chunking
and retrieval strategies across datasets, embedding models, and AI providers.

A typical user can:

1. Define datasets, strategy sets, embedding providers, LLM providers, and Optuna
   settings in YAML/JSON.
2. Run one experiment command that expands the matrix, prepares corpora when
   needed, runs HPO where configured, runs benchmarks, and writes structured
   artifacts.
3. Compare outputs through result JSON, manifest/log files, flat summaries, and
   leaderboard CSV files grouped by dataset/model/provider/strategy as needed.
4. Add a new dataset, provider, or strategy with a small local adapter instead of
   rewriting the benchmark pipeline.

## Non-Negotiable Boundaries

- The custom Dynamic Semantic chunking/expansion logic stays behaviorally intact.
  Infrastructure may wrap it, configure it, cache inputs, or pass optimized
  parameters into it, but must not redesign the algorithm.
- Built-in splitters and baselines are first-class comparison strategies, not
  hard-coded branches inside benchmark scripts.
- Provider support is configuration-driven and dry-run testable without requiring
  live API keys.
- Offline smoke tests must be possible with mock embeddings and static/custom
  datasets.
- Network/live provider checks are optional and explicit.

## Required Capabilities

### Strategies

- A registry exposes stable strategy ids, aliases, descriptions, and default
  parameters.
- Dynamic Semantic is registered as the focal custom strategy.
- Built-in alternatives include at least simple sentence/naive splitting, fixed
  windows, token splitting, markdown, HTML, JSON, and code-oriented splitting.
- Per-strategy parameter overrides can come from CLI flags, config files, or
  Optuna best-parameter artifacts.
- Benchmark outputs record both requested and effective strategy parameters.

### Datasets

- Static, QASPER, Wikipedia, and custom datasets are supported.
- Custom datasets can be loaded from JSONL and from separate articles/questions
  files.
- Dataset identity is preserved in result metadata and leaderboard grouping.
- Multi-dataset experiment runs can keep separate corpora and HPO artifacts per
  dataset.

### Embeddings

- Embedding providers are selected by config/CLI.
- Supported provider ids include mock, HuggingFace, Mistral, OpenAI, Ollama, and
  custom OpenAI-compatible endpoints.
- Provider catalogs expose default models, API-key environment variables, and
  base URLs where relevant.
- Dry-run provider validation reports missing keys/config without making live
  calls.

### LLM Providers

- LLM providers are selected by config/CLI where generation or provider smoke
  checks are needed.
- Supported provider ids include Mistral, OpenAI, OpenRouter, Ollama, and custom
  OpenAI-compatible endpoints.
- OpenRouter is documented and available out of the box through provider config.
- Live connectivity checks are explicit and separate from offline tests.

### Optuna

- HPO can be configured from YAML/JSON, including search spaces and objective
  policy.
- Cached HPO targets the Dynamic Semantic strategy; other strategies remain
  benchmark baselines unless a separate optimizer adapter is added.
- HPO runs can target different cached corpora/datasets independently.
- Best parameters and trial outputs are written to dataset-specific output
  directories.
- Benchmark runs can consume those best-parameter artifacts without manual
  translation.

### Experiment Runner

- A matrix runner expands dataset, embedding, LLM, strategy, and HPO axes.
- It supports validate-only and dry-run modes.
- It writes a manifest and logs for reproducibility.
- It supports pipeline-style runs where prepare, HPO, and benchmark phases are
  connected for each dataset.

### Outputs

- Benchmark result JSON contains enough metadata to identify source dataset,
  strategy id, provider/model choices, effective params, and metrics.
- Summary tools export JSONL/CSV and leaderboard CSV.
- Leaderboards can be grouped by dataset and other relevant axes.

### Project Hygiene

- Reusable implementation lives under `src/`.
- Root-level scripts are retained only for intentional CLI compatibility.
- Config examples live under `configs/`.
- Documentation explains setup, provider configuration, experiment configs,
  project structure, and target state.
- Runtime dependencies and dev dependencies are separated.
- The project declares supported Python metadata and passes local quality gates.

## Verification Gates

The target state is not complete until current-state evidence proves:

- `python run_benchmark.py --list-strategies` lists all core strategies.
- `python run_benchmark.py --list-providers` lists all embedding and LLM provider
  ids, including OpenRouter and custom endpoints.
- Offline benchmark smoke runs work with mock embeddings.
- Offline experiment smoke runs work for custom datasets and multi-dataset
  prepare + Optuna + benchmark pipelines.
- Provider dry-run checks work without API keys.
- `python run_smoke.py` runs the offline catalog/provider/config smoke checks.
- `pytest tests` passes.
- `ruff check .` passes.
- `python -m compileall` over the CLI scripts, `src/`, and `tests/` passes.
- `pip check` reports no broken requirements.
