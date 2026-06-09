# Extension Guide

Use this guide when adding a new dataset, provider, strategy, or HPO preset to
the benchmark stand. The goal is to keep extensions local and avoid rewriting
the pipeline scripts.

## Before Adding Code

Prefer configuration-only paths first:

- New benchmark data: use `--source custom` with `--dataset-path`, or paired
  `--articles-path` and `--questions-path`.
- OpenAI-compatible embedding endpoint: use `--embedding-provider custom` with
  `--embedding-base-url`.
- OpenAI-compatible chat endpoint: use `--llm-provider custom` with
  `--llm-base-url`.
- Different Dynamic Semantic HPO search space: add a YAML file under `configs/`
  and pass it through `--hpo-config`.

Add Python code only when the new behavior cannot be represented through those
config surfaces.

## Dataset Adapters

No code is needed for most pre-labeled datasets. A combined JSONL record can look
like this:

```json
{"title": "Doc", "text": "Alpha beta. Gamma delta.", "qa_pairs": [{"question": "What starts the doc?", "answer_sentence": "Alpha beta."}]}
```

Supported custom dataset loaders live in `src/benchmark_datasets.py`:

- combined JSON/JSONL articles with `qa_pairs`;
- JSON objects with `articles` and `questions`;
- paired article/question files compatible with `prepare_corpus.py` output.

When adding a new built-in source:

1. Add or update a loader under `src/`.
2. Add the source choice to `run_benchmark.py` and, if the source supports
   cached HPO, to `prepare_corpus.py`.
3. Preserve `dataset_name` metadata in benchmark outputs.
4. Add an offline fixture or smoke config when possible.
5. Add tests in `tests/test_infrastructure.py`.

Validation commands:

```bash
python run_benchmark.py --source custom --dataset-path data/custom_benchmark.jsonl --embedding-provider mock --embedding-model mock:64
python run_experiments.py configs/custom_smoke.yaml --validate-only
```

## Strategy Adapters

Strategies are first-class registry entries. Do not add ad hoc branches in the
benchmark loop.

To add a strategy:

1. Add a config dataclass in `src/config.py` and a `DEFAULT_*_CONFIG`.
2. Implement a `BaseStrategy` subclass in `src/strategies.py`.
3. Register stable ids, aliases, descriptions, override keys, and default params
   in `src/strategy_registry.py`.
4. Add the factory branch in `create_strategy`.
5. Add focused tests for aliases, catalog output, override validation, and a
   small smoke path.
6. Document the strategy in README/config examples if it is intended for users.

Dynamic Semantic remains the focal custom strategy. Wrapping, configuring, or
passing Optuna params into it is acceptable; changing its expansion algorithm is
outside the infrastructure scope.

Validation commands:

```bash
python run_benchmark.py --list-strategies
python run_benchmark.py --source static --embedding-provider mock --embedding-model mock:64 --strategies your_strategy_id
```

## Embedding Provider Adapters

Prefer `custom` for OpenAI-compatible endpoints. For a new first-class provider:

1. Add defaults to `DEFAULT_EMBEDDING_MODELS`, `DEFAULT_API_KEY_ENVS`, and, when
   needed, `DEFAULT_EMBEDDING_BASE_URLS` in `src/providers.py`.
2. Extend `build_embedding_model`.
3. Ensure `embedding_config_from_env` can validate missing required config
   without making network calls.
4. Update provider validation in `src/experiment_config.py` only if the provider
   has special requirements.
5. Add dry-run tests; keep live calls optional through `check_providers.py --run`.

Validation commands:

```bash
python run_benchmark.py --list-providers
python check_providers.py --embedding-provider mock --skip-llm
```

## LLM Provider Adapters

Prefer `openrouter` or `custom` for OpenAI-compatible chat endpoints. For a new
first-class provider:

1. Add defaults to `DEFAULT_CHAT_MODELS`, `DEFAULT_API_KEY_ENVS`, and
   `DEFAULT_CHAT_BASE_URLS` in `src/providers.py`.
2. Extend `build_llm` or `_openai_compatible_llm`.
3. Keep config validation dry-run safe.
4. Add tests that check catalog output and missing-config behavior without live
   API calls.

Validation commands:

```bash
python run_benchmark.py --list-providers
python check_providers.py --embedding-provider mock --llm-provider openrouter
```

## HPO Presets

HPO config files live under `configs/`. Cached HPO currently targets
`dynamic_semantic`; other strategies should remain benchmark baselines unless a
separate optimizer adapter is added.

To add a preset:

1. Create a YAML file with `search_space` and optional `objective`.
2. Keep parameter names aligned with `src/hpo_config.py` and
   `src/strategy_registry.py` override keys.
3. Add a validate-only or dry-run config that references the preset.
4. Add tests for config loading or experiment command expansion.

Validation commands:

```bash
python run_optuna.py --n-trials 1 --corpus-path results/corpora/my_dataset/cached_corpus.pkl --hpo-config configs/your_preset.yaml --no-viz
python run_experiments.py configs/custom_optuna_smoke.yaml --validate-only
```

## Required Gates

Run these after adding an adapter:

```bash
python -m pytest tests
python -m ruff check .
python -m compileall -q check_providers.py quick_benchmark.py run_benchmark.py run_comparison.py run_experiments.py run_optuna.py run_smoke.py summarize_results.py src tests
python -m pip check
```
