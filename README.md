# Dynamic Semantic Window for RAG [Experiment]

A learning experiment proving that **dynamic context expansion** based on cosine similarity of neighboring sentences provides cleaner, more relevant context (higher Signal-to-Noise Ratio) than standard chunking methods.

> **What if...** RAG context boundaries were determined *after* retrieval, expanding dynamically based on semantic similarity rather than fixed chunk sizes? Let's build it and find out.

## Table of Contents

- [Background](#background)
- [Hypothesis](#hypothesis)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)

## Background

Standard RAG (Retrieval-Augmented Generation) pipelines typically decide context boundaries **at indexing time**:

1. **Fixed Chunking** — Text is split into chunks of fixed size, often breaking mid-sentence or mid-thought
2. **Fixed Window** — Retrieved sentences are padded with a fixed number of neighbors (e.g., ±3 sentences)

Both methods have drawbacks: fixed chunking creates incomplete contexts, while fixed windows may include irrelevant information or miss semantically connected content.

This experiment introduces a **Dynamic Semantic Window** approach that determines context boundaries **at query time**, expanding retrieved sentences into semantically coherent clusters based on actual similarity between neighbors.

**Key difference**: Context boundaries are decided *after* the user query, allowing the system to adaptively include only the most relevant surrounding context.

## Hypothesis

> Dynamic context expansion using cosine similarity thresholds produces more coherent and relevant retrieval contexts than fixed-size chunking or fixed-window methods.

We compare three strategies:

| Strategy | Description |
|----------|-------------|
| **Baseline (Naive Chunking)** | `SentenceSplitter` with `chunk_size=256`, `overlap=20`. Top-k retrieval. |
| **Control (Fixed Window)** | `SentenceWindowNodeParser` with `window_size=3`. Fixed ±3 sentence padding. |
| **Experiment (Dynamic Semantic)** | Per-sentence indexing with greedy neighbor expansion while `cosine_similarity > threshold`. Merges overlapping clusters. |

## Installation

### Prerequisites

- Python 3.10+
- pip or uv package manager
- (Optional) GPU for local LLM inference via Ollama

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-dynamic-semantic-window.git
   cd rag-dynamic-semantic-window
   ```

2. Install dependencies:
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using uv (faster)
   uv pip install -r requirements.txt
   ```

3. Setup environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to set MISTRAL_API_KEY if evaluating answering quality
   ```

## Usage

### Run Benchmark

You can run the benchmark in two modes: **Static** (local file) or **Wikipedia** (random articles).

#### 1. Static Mode (Quantum Mechanics)
Runs benchmark on the included quantum mechanics text.
```bash
python run_benchmark.py --source=static
```

#### 2. Wikipedia Mode (Random or Specific)
Fetches articles from Wikipedia, generates QA pairs using Mistral LLM, and runs benchmark.

```bash
# Run on 5 random articles with 5 questions each
python run_benchmark.py --source=wikipedia --num-articles=5 --num-questions=5

# Run on a specific article
python run_benchmark.py --source=wikipedia --article="Graph theory" --num-questions=10
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Data source: `static` or `wikipedia` | `static` |
| `--num-articles` | Number of random articles to fetch | `1` |
| `--num-questions` | Questions per article | `10` |
| `--article` | Specific article title (overrides random) | `None` |

### Programmatic Usage

```python
from src.dynamic_retriever import DynamicSemanticExpander
from llama_index.core import VectorStoreIndex

# Create index with per-sentence nodes
index = create_sentence_index("data/source_text.txt")

# Apply dynamic expansion post-processor
expander = DynamicSemanticExpander(
    docstore=index.docstore,
    threshold=0.6,
    max_expand=5,
    min_window=1  # Always include ±1 neighbor
)

query_engine = index.as_query_engine(
    node_postprocessors=[expander]
)

response = query_engine.query("Your question here")
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MISTRAL_API_KEY` | Mistral API key for LLM-based evaluation | — |
| `SIMILARITY_THRESHOLD` | Cosine similarity threshold for expansion | `0.6` |
| `MAX_EXPAND` | Maximum sentences to expand in each direction | `5` |
| `TOP_K` | Number of sentences to retrieve initially | `5` |
| `EMBEDDING_MODEL` | HuggingFace embedding model name | `BAAI/bge-small-en-v1.5` |

## Project Structure

```
rag-dynamic-semantic-window/
├── data/
│   └── source_text.txt          # Test corpus
├── src/
│   ├── __init__.py
│   ├── dynamic_retriever.py     # Dynamic retrieval logic
│   ├── metrics.py               # IR metrics implementation
│   ├── question_generator.py    # LLM QA generation
│   ├── strategies.py            # Retrieval strategies
│   ├── wikipedia_loader.py      # Wiki data loader
│   └── utils.py                 # Utilities
├── run_benchmark.py             # Main benchmark script
├── requirements.txt
├── .env.example
├── CONCEPT_DESIGN.md            # Technical design document
└── README.md
```

## Evaluation Metrics

The benchmark evaluates retrieval quality using standard Information Retrieval metrics:

| Metric | Description |
|--------|-------------|
| **Hit Rate (HR@K)** | Fraction of questions where the answer is present in retrieved chunks. |
| **MRR** | Mean Reciprocal Rank — how high the first relevant chunk appears. |
| **Precision@K** | Fraction of retrieved chunks that contain the answer. |
| **Recall@K** | Fraction of relevant chunks retrieved (same as HR for single-answer). |
| **NDCG@K** | Normalized Discounted Cumulative Gain — accounts for rank position. |
| **Coherence** | Intra-cluster semantic similarity (higher = more coherent context). |
| **Token Count** | Total tokens retrieved (lower is better if HR is high). |

Results are exported to `results/benchmark_[timestamp].json`.

## License

MIT License
