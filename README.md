# Dynamic Semantic Window RAG

Experimental RAG retrieval benchmark for comparing fixed chunking with a
dynamic semantic window expansion strategy.

The project is organized around a small benchmark stand:

- prepare reusable dataset and embedding artifacts once;
- run comparable retrieval strategies against the same questions;
- save machine-readable results under `results/`.

## Background

Traditional RAG chunking uses fixed-size text windows. That is simple and often
strong for recall, but it can include redundant context or split semantically
connected passages.

This project tests `dynamic_semantic`: a retrieval strategy that starts from
seed sentences and expands context according to local semantic similarity,
query relevance, and expansion thresholds.

### Strategy Under Test

`dynamic_semantic` is designed to retrieve compact, semantically coherent
context. The current implementation focuses on:

- phantom embeddings computed with surrounding context;
- two-pass retrieval: broad candidate search followed by refined expansion;
- adaptive expansion based on local similarity signals;
- query-aware filtering while expanding neighboring sentences.

The main comparison point is whether this strategy can keep ranking quality high
while reducing tokens returned to the downstream LLM.

## Installation

### Requirements

- Python 3.12+
- Optional API keys for dataset QA generation or remote embedding/LLM providers
- Network access for fetching Wikipedia/QASPER data and downloading local
  Hugging Face embedding models when they are not cached

### Setup

```bash
git clone https://github.com/your-username/rag-dynamic-semantic-window.git
cd rag-dynamic-semantic-window

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Optional development tools
python -m pip install -r requirements-dev.txt
```

Install the local console command if desired:

```bash
python -m pip install -e .
```

## Usage

The benchmark stand has two stages.

### 1. Prepare Artifacts

Datasets are fetched and labeled once, then saved under
`artifacts/datasets/<name>/`. Embedding registrations and tuned params are also
saved under `artifacts/`.

```bash
# Wikipedia dataset with generated QA pairs
python -m stand prepare-dataset --source wikipedia --name wiki_100_qa \
  --num-articles 100 --min-length 6000 --questions-per-article 3 \
  --qa-provider openai --qa-model gpt-5.4-nano

# QASPER scientific papers
python -m stand prepare-dataset --source qasper --name qasper_val \
  --num-articles 30 --questions-per-article 3 --qa-provider mistral

# Pre-labeled custom JSONL dataset
python -m stand prepare-dataset --source custom --name my_set \
  --dataset-path data/custom_benchmark.jsonl

# Register or warm an embedding model
python -m stand prepare-embedding --name bge-base \
  --provider huggingface --model BAAI/bge-base-en-v1.5
```

Built-in embeddings:

- `mock`: offline smoke-test embeddings, not meaningful for retrieval quality;
- `bge-small`: local Hugging Face model `BAAI/bge-small-en-v1.5`.

Inspect available artifacts:

```bash
python -m stand list
```

### 2. Run Benchmarks

```bash
# Per-document retrieval: each article has its own index
python -m stand run --dataset wiki_100_qa --embedding bge-small \
  --index-mode per_document --params default

# Shared retrieval: all documents compete in one corpus-wide index
python -m stand run --dataset wiki_100_qa --embedding bge-small \
  --index-mode shared --params default

# Run only selected strategies
python -m stand run --dataset wiki_100_qa --embedding bge-small \
  --strategies fixed_window,dynamic_semantic --index-mode shared

# Use tuned dynamic_semantic params when an artifact exists
python -m stand run --dataset wiki_100_qa --embedding bge-small --params tuned
```

Index modes:

- `per_document`: chunks from each document are indexed separately; retrieval is
  scoped to the source document for each question. This isolates chunking and
  expansion behavior.
- `shared`: chunks from all documents go into one index. Retrieval competes
  across the whole corpus and is closer to production RAG.

Each run prints a metrics table and writes a result JSON under `results/`.

### Strategy IDs

- `naive`: LlamaIndex `SentenceSplitter`
- `fixed_window`: LlamaIndex `SentenceWindowNodeParser`
- `token_text`: LlamaIndex `TokenTextSplitter`
- `semantic_splitter`: LlamaIndex `SemanticSplitterNodeParser`
- `dynamic_semantic`: custom dynamic semantic window strategy

### Dataset Format

Combined JSONL, one document per line:

```jsonl
{"title": "Doc", "text": "Alpha beta. Gamma delta.", "qa_pairs": [{"question": "What starts the doc?", "answer": "Alpha beta", "answer_sentence": "Alpha beta."}]}
```

The evaluator uses exact normalized substring matching against
`answer_sentence`.

## Configuration

Create `.env` with the provider settings you use.

| Variable | Description | Required |
|----------|-------------|----------|
| `MISTRAL_API_KEY` | Mistral API key for QA generation or embeddings | If using Mistral |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI |
| `OPENROUTER_API_KEY` | OpenRouter API key | If using OpenRouter |
| `EMBEDDING_PROVIDER` | Default embedding provider id | No |
| `EMBEDDING_MODEL` | Default embedding model | No |
| `EMBEDDING_API_KEY_ENV` | Env var name containing the embedding API key | No |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embedding base URL | If custom |
| `LLM_PROVIDER` | Default QA-generation provider id | No |
| `LLM_MODEL` | Default QA-generation model | No |
| `LLM_API_KEY_ENV` | Env var name containing the LLM API key | No |
| `LLM_BASE_URL` | OpenAI-compatible LLM base URL | If custom |
| `QA_GENERATION_DELAY` | Delay between QA-generation calls | No |

Use `mock/mock:384` only for infrastructure smoke tests.

## Current Benchmark Results

Dataset: `wiki_100_qa`

- Source: English Wikipedia
- 100 articles
- 300 generated QA pairs
- QA model: `openai/gpt-5.4-nano`
- Embedding: `BAAI/bge-small-en-v1.5`
- `top_k=5`, `metric_k=5`
- Dynamic params: default

Result files:

- `results/benchmark_wiki_100_qa_per_document_20260704_012042.json`
- `results/benchmark_wiki_100_qa_shared_20260704_022407.json`

### Per-Document Index

Retrieval is scoped to the known source article for each question.

| Strategy | Tokens | HR@5 | MRR | P@5 | NDCG@5 |
|----------|-------:|-----:|----:|----:|-------:|
| Naive Chunking | 1207.9 | 0.9733 | 0.8294 | 0.2060 | 0.8648 |
| Fixed Window | 1153.4 | 0.9700 | 0.8525 | 0.3267 | 0.8601 |
| Token Text Splitter | 1353.7 | 0.9433 | 0.8004 | 0.1967 | 0.8358 |
| Semantic Splitter | 1176.8 | 0.9433 | 0.7733 | 0.1900 | 0.8159 |
| **Dynamic Semantic** | **614.5** | **0.9633** | **0.8847** | **0.1940** | **0.9050** |

### Shared Corpus Index

All chunks from all 100 documents compete in one index.

| Strategy | Tokens | HR@5 | MRR | P@5 | NDCG@5 |
|----------|-------:|-----:|----:|----:|-------:|
| Naive Chunking | 1208.9 | 0.9200 | 0.7888 | 0.1933 | 0.8214 |
| Fixed Window | 1100.6 | 0.9567 | 0.8375 | 0.2940 | 0.8510 |
| Token Text Splitter | 1347.3 | 0.9000 | 0.7599 | 0.1867 | 0.7948 |
| Semantic Splitter | 1136.0 | 0.9100 | 0.7363 | 0.1833 | 0.7799 |
| **Dynamic Semantic** | **627.2** | **0.9400** | **0.8494** | **0.1893** | **0.8727** |

## Development

Run tests and linting:

```bash
python -m pytest tests
python -m ruff check .
```

## License

MIT
