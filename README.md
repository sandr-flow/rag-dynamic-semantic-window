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

The benchmark stand has two layers:

- **Prep** (heavy, run once): build reusable artifacts on disk — datasets,
  embedding registrations, and per-domain tuned hyperparameters.
- **Run** (light): pick from prepared artifacts and benchmark one combination.

Install the console command once (optional):

```bash
python -m pip install -e .
```

### Interactive menu (primary entry point)

Run with no arguments to pick a prepared dataset, an embedding model, a strategy
set, an index mode, and (if available) tuned params from a console menu:

```bash
python -m stand
# or, if installed: stand
```

The menu only offers artifacts that already exist. Prepare them first with the
commands below.

### Prepare datasets and embeddings

Datasets are fetched/labeled once and reused. The QA-generation model is part of
a dataset's identity, so different QA models produce different datasets.

```bash
# QASPER scientific papers, questions generated with Mistral
python -m stand prepare-dataset --source qasper --name qasper_val \
  --num-articles 30 --questions-per-article 3 --qa-provider mistral

# Wikipedia, questions generated with OpenRouter
python -m stand prepare-dataset --source wikipedia --name wiki_30 \
  --num-articles 30 --min-length 6000 --qa-provider openrouter --qa-model openai/gpt-4.1-mini

# A pre-labeled custom dataset (no QA generation, no network)
python -m stand prepare-dataset --source custom --name my_set \
  --dataset-path data/custom_benchmark.jsonl

# Register an embedding model (downloads/warms HF weights, or validates an API model)
python -m stand prepare-embedding --name bge-base --provider huggingface --model BAAI/bge-base-en-v1.5
```

`mock` (offline, `mock:384`) and `bge-small` (local `BAAI/bge-small-en-v1.5`)
are always available without preparing anything.

### Tune the dynamic strategy per domain

The core hypothesis: `dynamic_semantic` hyperparameters should be tuned per
domain. Tuning is keyed by `(dataset, embedding)` and saved as a reusable
`tuned` artifact. The sentence/question embeddings are cached so re-tuning is fast.

```bash
python -m stand tune --dataset qasper_val --embedding bge-small --n-trials 200
```

### Run a benchmark non-interactively (CI / reproducibility)

The same in-process runner the menu uses, driven by flags:

```bash
# Offline smoke with mock embeddings
python -m stand run --dataset my_set --embedding mock --strategies naive,dynamic_semantic

# Compare per-document vs shared-corpus indexing
python -m stand run --dataset qasper_val --embedding bge-small --index-mode shared

# Apply the tuned dynamic params for this domain
python -m stand run --dataset qasper_val --embedding bge-small --params tuned
```

Index modes:

- `per_document` — chunks of each document go into their own collection; retrieval
  is scoped to that document.
- `shared` — chunks of all documents go into one collection; retrieval competes
  across the whole corpus (closer to real RAG, harder).

Each run prints a metrics table and saves a result JSON under `results/`.

### Inspect what is prepared

```bash
python -m stand list
```

### Strategy ids

- `naive` — LlamaIndex `SentenceSplitter`
- `fixed_window` — LlamaIndex `SentenceWindowNodeParser`
- `token_text` — LlamaIndex `TokenTextSplitter`
- `semantic_splitter` — LlamaIndex `SemanticSplitterNodeParser`
- `dynamic_semantic` — the custom dynamic semantic window strategy (the one under test)

Only text strategies are supported, so `default` and `all` are equivalent.

### Custom dataset format

Combined JSONL (one document per line):

```jsonl
{"title": "Doc", "text": "Alpha beta. Gamma delta.", "qa_pairs": [{"question": "What starts the doc?", "answer_sentence": "Alpha beta."}]}
```

### Quality gates

```bash
python -m pytest tests
python -m ruff check .
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

