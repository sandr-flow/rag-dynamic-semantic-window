# Experiment Configs

Configs are YAML files consumed by `run_experiments.py`.

- `static_smoke.yaml` runs a minimal static benchmark matrix.
- `custom_smoke.yaml` runs the checked-in custom dataset with mock embeddings.
- `dataset_matrix_smoke.yaml` runs the same strategy/provider setup across static and custom datasets.
- `custom_optuna_smoke.yaml` runs the full offline prepare -> Optuna -> benchmark chain.
- `multi_dataset_optuna_smoke.yaml` runs two independent prepare -> Optuna -> benchmark pipelines.
- `custom_optuna_example.yaml` demonstrates a full custom dataset workflow:
  prepare cached corpus, run Optuna, then benchmark with dataset-specific best params.
- `provider_matrix_example.yaml` demonstrates a custom dataset matrix over embedding
  providers, LLM providers, and strategy sets.
- `hpo_dynamic_balanced.yaml` defines Dynamic Semantic Optuna search space and objective weights.

When `optuna.apply_best_params: true`, `run_experiments.py` wires the Optuna JSON artifact
into the benchmark command. Prefer dataset-specific paths to avoid overwriting results:

```yaml
prepare_corpus:
  min_sentences: 1
  corpus_path: results/corpora/my_dataset/cached_corpus.pkl
  articles_output_path: results/corpora/my_dataset/articles.jsonl
  questions_output_path: results/corpora/my_dataset/questions.jsonl
optuna:
  corpus_path: results/corpora/my_dataset/cached_corpus.pkl
  output_dir: results/optuna/my_dataset
  best_params_json: results/optuna/my_dataset/best_params.json
  apply_best_params: true
```

Use dry-run before expensive model/API work:

```bash
python run_experiments.py configs/static_smoke.yaml --validate-only
python run_experiments.py configs/static_smoke.yaml --dry-run
python run_experiments.py configs/custom_optuna_example.yaml --dry-run
```

Benchmark configs support both retrieval depth and metric cutoff:

```yaml
benchmark:
  top_k: 10      # retrieve 10 chunks/clusters
  metric_k: 5    # report HR@5/P@5/R@5/NDCG@5
  qa_delay: 1.1  # delay between QA-generation LLM calls
```

Use `llms:` when QA generation should be compared across providers:

```yaml
llms:
  - provider: openrouter
    model: openai/gpt-4.1-mini
    api_key_env: OPENROUTER_API_KEY
  - provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
```

Use `datasets:` to compare the same strategy/provider matrix across multiple benchmark inputs:

```yaml
benchmark:
  top_k: 5
  metric_k: 5
datasets:
  - name: static
    source: static
  - name: custom_sample
    source: custom
    dataset_path: data/custom_benchmark.jsonl
```

Use `pipelines:` when each dataset needs its own corpus cache and Optuna artifacts:

```yaml
pipelines:
  - name: dataset_a
    prepare_corpus: {enabled: true, source: custom, corpus_path: results/corpora/a.pkl}
    optuna: {enabled: true, corpus_path: results/corpora/a.pkl, output_dir: results/optuna/a}
    benchmark: {enabled: true, source: custom, dataset_name: dataset_a}
```
