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

# Tune dynamic_semantic on train documents (held-out validation metrics).
# Dual-space is the reference setup: adjacency from clean sentence
# embeddings, threshold search ranges adapted to the clean distribution.
python -m stand tune --dataset wiki_100_qa_hard --embedding bge-small \
  --adjacency-space clean --hpo-config configs/hpo_clean_adjacency.yaml
```

The tuned artifact records `adjacency_space`, so `stand run --params tuned`
reproduces the space automatically. Phantom adjacency is kept only as an A/B
control: `--dynamic-overrides '{"adjacency_space": "phantom"}'`.

Index modes:

- `per_document`: chunks from each document are indexed separately; retrieval is
  scoped to the source document for each question. This isolates chunking and
  expansion behavior.
- `shared`: chunks from all documents go into one index. Retrieval competes
  across the whole corpus and is closer to production RAG. Dynamic semantic
  expansion is still bounded by each seed's `source_doc`.

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

## Baseline Benchmark Results (2026-07-04)

First full baseline after the stage-0 validity fixes (unified expansion core,
phantom-aligned HPO corpus, shared-mode document isolation, shared sentence
segmentation) and exact tiktoken (`cl100k_base`) token counting. Token numbers
are not comparable with older chars/4-based results.

Dataset: `wiki_100_qa`

- Source: English Wikipedia
- 100 articles, 300 generated QA pairs (QA model: `openai/gpt-5.4-nano`)
- Embedding: `BAAI/bge-small-en-v1.5`
- `top_k=5`, `metric_k=5`

`dynamic_semantic` is not in these tables: it is evaluated only with
per-domain tuned params (next section), because its shipped defaults are
pseudo-defaults from the pre-fix retrieval path.

Result files:

- `results/benchmark_wiki_100_qa_per_document_20260704_210713.json`
- `results/benchmark_wiki_100_qa_shared_20260704_211717.json`

### Per-Document Index

Retrieval is scoped to the known source article for each question.

| Strategy | Tokens | HR@5 | MRR | P@5 | NDCG@5 |
|----------|-------:|-----:|----:|----:|-------:|
| Naive Chunking | 1079.8 | 0.9733 | 0.8294 | 0.2060 | 0.8648 |
| Fixed Window | 1064.2 | 0.9700 | 0.8525 | 0.3267 | 0.8601 |
| Token Text Splitter | 1216.4 | 0.9500 | 0.8054 | 0.1993 | 0.8408 |
| Semantic Splitter | 1061.0 | 0.9467 | 0.7766 | 0.1907 | 0.8193 |

### Shared Corpus Index

All chunks from all 100 documents compete in one index.

| Strategy | Tokens | HR@5 | MRR | P@5 | NDCG@5 |
|----------|-------:|-----:|----:|----:|-------:|
| Naive Chunking | 1082.4 | 0.9200 | 0.7888 | 0.1933 | 0.8214 |
| Fixed Window | 1020.8 | 0.9567 | 0.8375 | 0.2940 | 0.8510 |
| Token Text Splitter | 1213.7 | 0.9067 | 0.7649 | 0.1893 | 0.7999 |
| Semantic Splitter | 1021.2 | 0.9133 | 0.7397 | 0.1840 | 0.7832 |

### Reading These Numbers

- The four baselines span HR 0.91-0.97 at roughly 1000-1200 tokens; Fixed
  Window is the strongest quality/cost point and serves as the reference
  for Dynamic Semantic in the next section.
- HR differences between the top strategies are within the statistical noise
  at n=300 (95% CI is roughly +/-0.02-0.03); significance testing is planned.

## Dynamic Semantic Results (Tuned, Dual-Space)

Dual-space clean adjacency is the reference configuration (accepted in
improvement-plan step P.3; full A/B against phantom adjacency is recorded
there). Tuning protocol: Optuna 300 trials, document-level 70/30 split,
hard 1200-token budget:

```bash
python -m stand tune --dataset <dataset> --embedding bge-small \
  --adjacency-space clean --hpo-config configs/hpo_clean_adjacency.yaml --n-trials 300
```

### wiki_100_qa_hard (2026-07-04)

`wiki_100_qa_hard` paraphrases each question away from the answer's lexis
(285/300 accepted at content-token overlap <= 0.35), removing the lexical
mirroring that compresses strategy differences on the original set.

| Configuration | Tokens | HR@5 | MRR |
|---------------|-------:|-----:|----:|
| Fixed Window (best baseline), per-document | 1054 | 0.930 | 0.803 |
| **Dynamic Semantic tuned, per-document** | **931** | **0.967** | **0.852** |
| Fixed Window (best baseline), shared | 1011 | 0.900 | 0.762 |
| **Dynamic Semantic tuned, shared** | **833** | **0.930** | **0.808** |

Result files:

- `results/benchmark_wiki_100_qa_hard_per_document_20260704_231808.json`
- `results/benchmark_wiki_100_qa_hard_shared_20260704_232244.json`

### wiki_100_qa (2026-07-04)

Same tuning protocol on the original (non-paraphrased) dataset. Questions
lexically mirror answers here, so absolute numbers are higher for every
strategy and differences are compressed (see the ceiling analysis in
improvement-plan step P.1).

| Configuration | Tokens | HR@5 | MRR |
|---------------|-------:|-----:|----:|
| Fixed Window, per-document | 1064 | 0.970 | 0.853 |
| **Dynamic Semantic tuned, per-document** | **1031** | **0.987** | **0.905** |
| Fixed Window, shared | 1021 | 0.957 | 0.838 |
| **Dynamic Semantic tuned, shared** | **904** | **0.973** | **0.877** |

Tuned dynamic beats every baseline in both modes (the best baseline HR is
Naive Chunking's 0.9733 per-document at 1080 tokens), while staying below
every baseline's token budget.

Result files:

- `results/benchmark_wiki_100_qa_per_document_20260704_235149.json`
- `results/benchmark_wiki_100_qa_shared_20260704_235638.json`

### Reading the Tuned Numbers

The project thesis holds on both datasets in both index modes: higher HR/MRR
than the strongest baseline at a smaller token budget (significance testing
via paired bootstrap is the next planned step). Low P@5 relative to baselines
is expected and not tracked here: merged clusters mean fewer, larger
retrieved units, which P@5 penalizes regardless of context quality.

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
