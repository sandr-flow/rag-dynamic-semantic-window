# Dynamic Semantic Window RAG

Experimental RAG retrieval strategy using dynamic context window expansion based on semantic similarity.

> **Active Experiment**: This project is under active development. Results and configurations may change.

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Benchmark Results](#benchmark-results)
- [License](#license)

## Background

Traditional RAG chunking methods use fixed-size windows, which can split related context or include irrelevant text. This project explores **dynamic semantic window expansion**: starting from a seed sentence and expanding the context window based on cosine similarity with neighboring sentences.

### Key Features

- **Phantom Embeddings**: Embeddings computed using surrounding context for better semantic representation
- **Two-Pass Retrieval**: Broad initial search (`top_k * multiplier`), then refined expansion
- **Adaptive Thresholds**: Dynamic expansion based on local density and gradient detection
- **Query-Aware Expansion**: Considers both neighbor similarity and query relevance

## Installation

### Prerequisites

- Python 3.12+ recommended; the current local venv was recreated with Python 3.14.4
- API key for QA generation when using remote LLM providers

### Steps

```bash
# Clone repository
git clone https://github.com/your-username/rag-dynamic-semantic-window.git
cd rag-dynamic-semantic-window

# Create virtual environment
py -3.14 -m venv .venv  # Windows, when Python 3.14 is installed
# python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Optional: install local test/lint tools
python -m pip install -r requirements-dev.txt
```

## Usage

### Quick Interactive Benchmark

For manual runs, start the wrapper without arguments and choose dataset, embedding
model, and strategies after launch:

```bash
python quick_benchmark.py
```

For a one-line run on an existing custom dataset:

```bash
python quick_benchmark.py data/custom_benchmark.jsonl --provider mock --model mock:384
```

To print the generated low-level command without running it:

```bash
python quick_benchmark.py data/custom_benchmark.jsonl --provider mock --model mock:384 --print-command
```

### Run Benchmark

```bash
# Inspect available strategies and providers
python run_benchmark.py --list-strategies
python run_benchmark.py --list-providers

# Validate provider config without network calls
python check_providers.py --embedding-provider mock --llm-provider openrouter

# Run a minimal live provider smoke check
python check_providers.py --run --embedding-provider openai --llm-provider openrouter

# Static smoke benchmark
python run_benchmark.py --source static --embedding-provider mock --embedding-model mock:384 --strategies naive,fixed_window,token_text,dynamic_semantic

# Wikipedia articles
python run_benchmark.py --source wikipedia --num-articles 30 --num-questions 3 --min-length 6000

# QASPER scientific papers
python run_benchmark.py --source qasper --num-articles 30 --num-questions 3 --min-length 4000

# Use OpenRouter for QA generation
python run_benchmark.py --source wikipedia --llm-provider openrouter --llm-model openai/gpt-4.1-mini

# Use a different embedding model
python run_benchmark.py --source static --embedding-provider huggingface --embedding-model BAAI/bge-base-en-v1.5

# Use a custom pre-labeled dataset
python run_benchmark.py --source custom --dataset-path path/to/dataset.jsonl

# Or reuse prepare_corpus-style paired files
python run_benchmark.py --source custom --articles-path data/articles.jsonl --questions-path data/questions.jsonl
```

`--list-strategies` prints stable strategy ids, default/optional status, supported
override keys, and current default parameters for each strategy.

### Run Experiment Matrix

Use YAML/JSON configs for repeatable matrix runs over strategies and embedding models:

```bash
# Inspect generated commands without running model downloads or API calls
python run_experiments.py configs/static_smoke.yaml --dry-run

# Validate config expansion without writing logs/manifests
python run_experiments.py configs/static_smoke.yaml --validate-only

# Run the configured matrix and save logs/manifest under results/experiments/
python run_experiments.py configs/static_smoke.yaml

# Run the checked-in custom dataset without model downloads/API calls
python run_experiments.py configs/custom_smoke.yaml

# Run the full offline prepare -> Optuna -> benchmark smoke
python run_experiments.py configs/custom_optuna_smoke.yaml

# Run independent dataset-specific prepare -> Optuna -> benchmark pipelines
python run_experiments.py configs/multi_dataset_optuna_smoke.yaml --dry-run

# Inspect a provider matrix over embeddings, LLM providers, and strategy sets
python run_experiments.py configs/provider_matrix_example.yaml --dry-run

# Run offline smoke checks for catalogs, providers, and experiment config expansion
python run_smoke.py

# Run full offline custom + Optuna smoke pipelines
python run_smoke.py --full
```

When benchmark commands run through `run_experiments.py`, the runner records newly created
`results/benchmark_*.json` files in the manifest and writes:

- `benchmark_summary.csv`
- `benchmark_summary.jsonl`
- `benchmark_leaderboard.csv`
- `benchmark_leaderboard_by_dataset.csv`

Summary rows include dataset, embedding, LLM, strategy, metric cutoff, question/article
counts, and averaged metrics. `llm_used_for_qa_generation` marks whether the LLM was
actually used to generate QA pairs or only recorded as a configured matrix axis.
The manifest also stores a config hash, runner metadata, generated command count,
per-command logs, and expected output artifacts for prepare, Optuna, and benchmark phases.

You can also summarize benchmark files manually:

```bash
python summarize_results.py
python summarize_results.py results/benchmark_20260119_233351.json --csv results/one_run.csv
python summarize_results.py results/benchmark_20260119_233351.json --leaderboard-csv results/one_run_leaderboard.csv
```

### Quality Gates

```bash
python run_smoke.py --quality
python -m pytest tests
python -m ruff check .
```

Minimal config shape:

```yaml
name: static_smoke
benchmark:
  enabled: true
  source: static
  top_k: 5
  metric_k: 5
  qa_delay: 1.1
embeddings:
  - provider: huggingface
    model: BAAI/bge-small-en-v1.5
strategy_sets:
  - strategies: [naive, fixed_window, token_text, semantic_splitter, dynamic_semantic]
```

Use `datasets:`, `embeddings:`, `llms:`, and `strategy_sets:` lists to create benchmark
matrices. See `configs/dataset_matrix_smoke.yaml` for a dataset axis and
`configs/provider_matrix_example.yaml` for a custom dataset example that includes OpenRouter.
Use `pipelines:` when each dataset needs its own prepared corpus and Optuna artifacts.
See `docs/extension_guide.md` before adding a new dataset, provider, strategy, or HPO preset.

### Prepare Corpus and Optimize Dynamic Strategy

```bash
# Build a cached corpus for fast Optuna trials
python prepare_corpus.py --source qasper --num-articles 30 --questions-per-article 3 --top-k 100

# Or build a cached corpus from pre-labeled custom data without QA-generation calls
python prepare_corpus.py --source custom --dataset-path path/to/dataset.jsonl --top-k 100

# Keep cached corpora separate per dataset/embedding setup
python prepare_corpus.py --source custom --dataset-path path/to/dataset.jsonl \
  --min-sentences 1 \
  --corpus-path results/corpora/my_dataset/cached_corpus.pkl \
  --articles-output-path results/corpora/my_dataset/articles.jsonl \
  --questions-output-path results/corpora/my_dataset/questions.jsonl

# Optimize Dynamic Semantic hyperparameters on that corpus
python run_optuna.py --n-trials 200 --corpus-path results/corpora/my_dataset/cached_corpus.pkl --target-clusters 5 --soft-token-limit 1200

# Use explicit search space and scoring policy
python run_optuna.py --n-trials 200 --corpus-path data/cached_corpus.pkl --hpo-config configs/hpo_dynamic_balanced.yaml

# Save HPO artifacts under a dataset-specific directory
python run_optuna.py --n-trials 200 --corpus-path data/cached_corpus.pkl --output-dir results/optuna/qasper

# Re-run benchmark with the optimized dynamic params
python run_benchmark.py --source qasper --strategy-overrides-json results/optuna/qasper/best_params.json
```

HPO config files can override the cached Dynamic Semantic search space and objective policy:

```yaml
search_space:
  threshold: {type: float, low: 0.5, high: 0.99, step: 0.001}
  max_expand: {type: int, low: 3, high: 10}
objective:
  soft_token_limit: 1200
  hr_weight: 100.0
  mrr_weight: 10.0
  token_bonus_weight: 5.0
  token_penalty_per_token: 0.01
```

`--strategy-overrides-json` accepts either flat Optuna params:

```json
{"threshold": 0.91, "max_expand": 4}
```

Flat params are treated as legacy Dynamic Semantic overrides and are applied only to
`dynamic_semantic`. Use grouped params for other strategies or shared settings.

or grouped per-strategy params:

```json
{
  "naive": {"chunk_size": 256, "chunk_overlap": 40},
  "token_text": {"chunk_size": 256, "chunk_overlap": 40},
  "dynamic_semantic": {"threshold": 0.91, "max_expand": 4}
}
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Data source: `static`, `wikipedia`, `qasper`, `custom` | `static` |
| `--num-articles` | Number of articles to benchmark | `5` |
| `--num-questions` | Questions per article | `3` |
| `--min-length` | Minimum article length (chars) | `2000` |
| `--strategies` | Comma-separated strategy ids | `default` |
| `--top-k` | Number of retrieved chunks/clusters | `5` |
| `--metric-k` | Cutoff for HR/P/R/NDCG metrics; defaults to `--top-k` | `None` |
| `--strategy-overrides-json` | JSON params, including Optuna `best_params.json` | `None` |
| `--embedding-provider` | `mock`, `huggingface`, `mistral`, `openai`, `ollama`, `custom` | env/default |
| `--embedding-model` | Embedding model override | env/default |
| `--llm-provider` | `mistral`, `openai`, `openrouter`, `ollama`, `custom` | env/default |
| `--llm-model` | Chat model override for QA generation | env/default |
| `--qa-delay` | Delay between QA-generation LLM calls in seconds | env/default |
| `--dataset-path` | Combined JSON/JSONL custom dataset | `None` |
| `--dataset-name` | Label stored in result metadata for reports | `None` |
| `--articles-path` | Articles JSON/JSONL for paired custom dataset | `None` |
| `--questions-path` | Questions JSON/JSONL for paired custom dataset | `None` |

Strategy ids:

- `naive` - LlamaIndex `SentenceSplitter`
- `fixed_window` - LlamaIndex `SentenceWindowNodeParser`
- `token_text` - LlamaIndex `TokenTextSplitter`
- `semantic_splitter` - LlamaIndex `SemanticSplitterNodeParser`
- `dynamic_semantic` - the custom dynamic semantic window strategy
- `markdown` - LlamaIndex `MarkdownNodeParser`
- `html` - LlamaIndex `HTMLNodeParser`
- `json` - LlamaIndex `JSONNodeParser`
- `code` - LlamaIndex `CodeSplitter`

`default` uses the general text strategies. `all` includes format-specific parsers too:

```bash
python run_benchmark.py --source custom --dataset-path path/to/markdown_dataset.jsonl --strategies markdown,dynamic_semantic
python run_benchmark.py --source custom --dataset-path path/to/code_dataset.jsonl --strategies code,token_text --strategy-overrides-json code_params.json
```

### Custom Dataset Format

Combined JSONL:

```jsonl
{"title": "Doc", "text": "Alpha beta. Gamma delta.", "qa_pairs": [{"question": "What starts the doc?", "answer_sentence": "Alpha beta."}]}
```

Paired files, compatible with `prepare_corpus.py` outputs:

```jsonl
{"id": 10, "title": "Doc", "text": "Alpha beta. Gamma delta."}
```

```jsonl
{"article_id": 10, "question": "What starts the doc?", "answer_sentence": "Alpha beta."}
```

## Configuration

Create `.env` file based on `.env.example`:

| Variable | Description | Required |
|----------|-------------|----------|
| `MISTRAL_API_KEY` | Mistral API key for QA generation or embeddings | If using Mistral |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI |
| `OPENROUTER_API_KEY` | OpenRouter API key | If using OpenRouter |
| `EMBEDDING_PROVIDER` | Embedding provider id | No |
| `EMBEDDING_MODEL` | Embedding model | No |
| `EMBEDDING_BASE_URL` | OpenAI-compatible base URL for custom embedding providers | If custom embedding |
| `LLM_PROVIDER` | QA-generation provider id | No |
| `LLM_MODEL` | QA-generation model | No |
| `LLM_BASE_URL` | OpenAI-compatible base URL for custom providers | No |
| `QA_GENERATION_DELAY` | Delay between QA-generation provider calls | No |

Use `EMBEDDING_PROVIDER=mock` and `EMBEDDING_MODEL=mock:384` only for infrastructure smoke
tests. It avoids model downloads/API calls but does not produce meaningful retrieval quality metrics.

## Benchmark Results

### Wikipedia (2026-01-20)

199 articles, 581 questions, min_length=6000

| Strategy | Tokens | HR@5 | MRR | NDCG |
|----------|--------|------|-----|------|
| Naive Chunking | 588 | 0.91 | 0.74 | 0.79 |
| Fixed Window | 1158 | 0.93 | 0.75 | 0.78 |
| Semantic Splitter | 1226 | 0.92 | 0.74 | 0.79 |
| **Dynamic Semantic** | **655** | **0.95** | **0.87** | **0.89** |

### QASPER (2026-01-20)

30 articles, 89 questions, min_length=4000

| Strategy | Tokens | HR@5 | MRR | NDCG |
|----------|--------|------|-----|------|
| Naive Chunking | 667 | 0.67 | 0.50 | 0.54 |
| Fixed Window | 1336 | 0.85 | 0.65 | 0.68 |
| Semantic Splitter | 1390 | 0.69 | 0.46 | 0.51 |
| **Dynamic Semantic** | **766** | **0.75** | **0.65** | **0.68** |

### Key Observations

- **Wikipedia**: Dynamic Semantic achieves best MRR (0.87) with 15% fewer tokens than Fixed Window
- **QASPER**: Dynamic Semantic matches Fixed Window MRR with 43% fewer tokens
- Scientific papers are more challenging due to technical terminology and complex structure

## License

MIT
