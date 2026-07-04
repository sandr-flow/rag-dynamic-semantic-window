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

- dual-space embeddings: phantom embeddings (sentence + neighbors) for
  matching against the query, clean per-sentence embeddings for deciding
  where the topic ends and expansion must stop;
- two-pass retrieval: broad candidate search followed by refined expansion;
- adaptive expansion driven by adjacency similarity in the clean space;
- query-aware filtering while expanding neighboring sentences.

Parameters are tuned per domain (`stand tune`). The strategy is always
evaluated with tuned params: the values shipped in `src/config.py` are
legacy artifacts of the pre-fix retrieval path (pseudo-defaults), not a
meaningful configuration.

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
# Shared index (default, primary mode): all documents compete in one
# corpus-wide index; dynamic_semantic uses per-domain tuned params
python -m stand run --dataset wiki_100_qa_hard --embedding bge-small --params tuned

# Per-document index (diagnostic): retrieval scoped to the source article
python -m stand run --dataset wiki_100_qa_hard --embedding bge-small \
  --index-mode per_document --params tuned

# Run only selected strategies
python -m stand run --dataset wiki_100_qa_hard --embedding bge-small \
  --strategies fixed_window,dynamic_semantic

# Tune dynamic_semantic on train documents (held-out validation metrics).
# The expansion adjacency signal always comes from clean sentence
# embeddings; default search ranges are calibrated to that distribution.
python -m stand tune --dataset wiki_100_qa_hard --embedding bge-small

# Paired bootstrap CIs (dHR/dMRR vs baselines) for saved results; baseline
# rows may come from a second file of the same dataset + index mode
python -m stand significance results/benchmark_x_shared_A.json \
  --baseline-result results/benchmark_x_shared_B.json
```

Index modes:

- `shared` (default, primary): chunks from all documents go into one index.
  Retrieval competes across the whole corpus, as in production RAG. Dynamic
  semantic expansion is still bounded by each seed's `source_doc`.
- `per_document` (diagnostic): chunks from each document are indexed
  separately; retrieval is scoped to the source document for each question.
  Isolates chunking and expansion behavior from cross-document competition.

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

The evaluator checks retrieved chunks against `answer_sentence` with exact
normalized containment, plus a contiguous-token-window fallback that only
tolerates sentences truncated at chunk boundaries. Fuzzy matching is used only
to resolve the ground-truth sentence index when building HPO corpora.

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
| `EMBEDDING_BATCH_SIZE` | Batch size for embedding calls (default 64) | No |
| `EMBEDDING_CACHE` | Set `0` to disable the shared on-disk embedding cache | No |
| `LLM_PROVIDER` | Default QA-generation provider id | No |
| `LLM_MODEL` | Default QA-generation model | No |
| `LLM_API_KEY_ENV` | Env var name containing the LLM API key | No |
| `LLM_BASE_URL` | OpenAI-compatible LLM base URL | If custom |
| `QA_GENERATION_DELAY` | Delay between QA-generation calls | No |

Use `mock/mock:384` only for infrastructure smoke tests.

## Benchmark Results (2026-07-04)

First full results after the stage-0 validity fixes (unified expansion core,
phantom-aligned HPO corpus, shared-mode document isolation, shared sentence
segmentation) and exact tiktoken (`cl100k_base`) token counting. Token numbers
are not comparable with older chars/4-based results.

Both datasets share the same 100 English Wikipedia articles and 300 generated
QA pairs (QA model: `openai/gpt-5.4-nano`). `wiki_100_qa_hard` additionally
paraphrases each question away from the answer's lexis (285/300 accepted at
content-token overlap <= 0.35) and is the primary dataset: the original's
lexical mirroring compresses strategy differences (ceiling analysis in
improvement-plan step P.1).

Setup: `BAAI/bge-small-en-v1.5`, `top_k=5`, `metric_k=5`. Baselines run
library-default params. Dynamic Semantic runs per-domain tuned params
(`python -m stand tune`: Optuna 300 trials, document-level 70/30 split, hard
1200-token budget); its shipped defaults are pseudo-defaults from the
pre-fix retrieval path and are not benchmarked.

**The shared index is the primary evaluation mode**: chunks from all
documents compete in one corpus-wide index, as in production RAG.
Per-document results (retrieval scoped to the known source article) are kept
as a diagnostic that isolates chunking behavior from cross-document
competition.

### wiki_100_qa_hard — shared index (primary)

| Strategy | Tokens | HR@5 | MRR | P@5 | NDCG@5 |
|----------|-------:|-----:|----:|----:|-------:|
| Naive Chunking | 1091.7 | 0.8800 | 0.7007 | 0.1840 | 0.7453 |
| Fixed Window | 1010.9 | 0.9000 | 0.7622 | 0.2600 | 0.7850 |
| Token Text Splitter | 1213.9 | 0.8500 | 0.6689 | 0.1767 | 0.7143 |
| Semantic Splitter | 970.2 | 0.8267 | 0.6549 | 0.1660 | 0.6983 |
| **Dynamic Semantic (tuned)** | **832.9** | **0.9300** | **0.8084** | 0.1873 | **0.8391** |

### wiki_100_qa_hard — per-document index (diagnostic)

| Strategy | Tokens | HR@5 | MRR | P@5 | NDCG@5 |
|----------|-------:|-----:|----:|----:|-------:|
| Naive Chunking | 1088.1 | 0.9533 | 0.7698 | 0.2007 | 0.8147 |
| Fixed Window | 1054.3 | 0.9300 | 0.8025 | 0.3147 | 0.8154 |
| Token Text Splitter | 1214.6 | 0.9367 | 0.7505 | 0.1960 | 0.7977 |
| Semantic Splitter | 1034.8 | 0.9200 | 0.7170 | 0.1847 | 0.7680 |
| **Dynamic Semantic (tuned)** | **930.6** | **0.9667** | **0.8524** | 0.1947 | **0.8816** |

Result files: baselines
`results/benchmark_wiki_100_qa_hard_{shared_20260704_222345,per_document_20260704_221258}.json`,
tuned dynamic
`results/benchmark_wiki_100_qa_hard_{shared_20260704_232244,per_document_20260704_231808}.json`.

### wiki_100_qa — shared index (primary)

| Strategy | Tokens | HR@5 | MRR | P@5 | NDCG@5 |
|----------|-------:|-----:|----:|----:|-------:|
| Naive Chunking | 1082.4 | 0.9200 | 0.7888 | 0.1933 | 0.8214 |
| Fixed Window | 1020.8 | 0.9567 | 0.8375 | 0.2940 | 0.8510 |
| Token Text Splitter | 1213.7 | 0.9067 | 0.7649 | 0.1893 | 0.7999 |
| Semantic Splitter | 1021.2 | 0.9133 | 0.7397 | 0.1840 | 0.7832 |
| **Dynamic Semantic (tuned)** | **903.9** | **0.9733** | **0.8771** | 0.1960 | **0.9018** |

### wiki_100_qa — per-document index (diagnostic)

| Strategy | Tokens | HR@5 | MRR | P@5 | NDCG@5 |
|----------|-------:|-----:|----:|----:|-------:|
| Naive Chunking | 1079.8 | 0.9733 | 0.8294 | 0.2060 | 0.8648 |
| Fixed Window | 1064.2 | 0.9700 | 0.8525 | 0.3267 | 0.8601 |
| Token Text Splitter | 1216.4 | 0.9500 | 0.8054 | 0.1993 | 0.8408 |
| Semantic Splitter | 1061.0 | 0.9467 | 0.7766 | 0.1907 | 0.8193 |
| **Dynamic Semantic (tuned)** | **1030.6** | **0.9867** | **0.9053** | 0.1987 | **0.9263** |

Result files: baselines
`results/benchmark_wiki_100_qa_{shared_20260704_211717,per_document_20260704_210713}.json`,
tuned dynamic
`results/benchmark_wiki_100_qa_{shared_20260704_235638,per_document_20260704_235149}.json`.

### Reading These Numbers

- In the primary (shared) mode Dynamic Semantic tops every metric except
  P@5 on both datasets while spending fewer tokens than every baseline.
- Paired bootstrap over questions (10000 resamples, 95% CI;
  `stand significance`): in shared mode ΔHR and ΔMRR vs Naive, Token Text,
  and Semantic Splitter are significantly positive on both datasets; vs
  Fixed Window ΔMRR is significantly positive (+0.046 hard, +0.040 wiki)
  and ΔHR is positive but within noise (+0.030 / +0.017) at ~18%/11% fewer
  tokens. In the per-document diagnostic ΔHR vs Fixed Window is significant
  on hard (+0.037 [+0.007, +0.067]). HR is never significantly below any
  baseline anywhere. Full CI tables: `docs/improvement_plan.md`, step 1.2.
- Low P@5 for Dynamic Semantic is expected: merged clusters mean fewer,
  larger retrieved units, which P@5 penalizes regardless of context quality.

## Legacy Benchmark Results (Outdated)

The numbers below are kept only as historical context. They were produced
before the stage-0 validity fixes: unified expansion core, phantom-aligned HPO
corpus, shared-mode document isolation, shared sentence segmentation, and
held-out tuning metrics. Token counts used the rough chars/4 estimate. Do not
use them as the current baseline.

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
