# Dynamic Semantic Window for RAG

A learning experiment proving that **dynamic context expansion** based on cosine similarity of neighboring sentences provides cleaner, more relevant context (higher Signal-to-Noise Ratio) than standard chunking methods.

> Part of the **"What if..."** experimental series.

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

Standard RAG (Retrieval-Augmented Generation) pipelines typically use one of two approaches:

1. **Fixed Chunking** — Text is split into chunks of fixed size, often breaking mid-sentence or mid-thought
2. **Fixed Window** — Retrieved sentences are padded with a fixed number of neighbors (e.g., ±3 sentences)

Both methods have drawbacks: fixed chunking creates incomplete contexts, while fixed windows may include irrelevant information or miss semantically connected content.

This experiment introduces a **Dynamic Semantic Window** approach that expands context boundaries based on actual semantic similarity between neighboring sentences.

## Hypothesis

> Dynamic context expansion using cosine similarity thresholds produces more coherent and relevant retrieval contexts than fixed-size chunking or fixed-window methods.

We compare three strategies:

| Strategy | Description |
|----------|-------------|
| **Baseline (Naive Chunking)** | `SentenceSplitter` with `chunk_size=256`, `overlap=20`. Top-k retrieval. |
| **Control (Fixed Window)** | `SentenceWindowNodeParser` with `window_size=3`. Fixed ±3 sentence padding. |
| **Experiment (Dynamic Semantic)** | Per-sentence indexing with greedy neighbor expansion while `cosine_similarity > threshold`. |

## Installation

### Prerequisites

- Python 3.10+
- pip or uv package manager
- (Optional) GPU for local LLM inference via Ollama

### Steps

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rag-dynamic-semantic-window.git
cd rag-dynamic-semantic-window
```

2. Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/macOS
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables (see [Configuration](#configuration)):

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Run Benchmark

Execute the full comparison benchmark:

```bash
python run_benchmark.py
```

### Interactive Demo

Explore the results in Jupyter:

```bash
jupyter notebook notebook_demo.ipynb
```

### Programmatic Usage

```python
from src.dynamic_retriever import DynamicSemanticExpander
from llama_index.core import VectorStoreIndex

# Create index with per-sentence nodes
index = create_sentence_index("data/source_text.txt")

# Apply dynamic expansion post-processor
expander = DynamicSemanticExpander(
    docstore=index.docstore,
    threshold=0.75,
    max_expand=5
)

query_engine = index.as_query_engine(
    node_postprocessors=[expander]
)

response = query_engine.query("Your question here")
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-based evaluation | — |
| `SIMILARITY_THRESHOLD` | Cosine similarity threshold for expansion | `0.75` |
| `MAX_EXPAND` | Maximum sentences to expand in each direction | `5` |
| `TOP_K` | Number of sentences to retrieve initially | `5` |
| `EMBEDDING_MODEL` | HuggingFace embedding model name | `BAAI/bge-small-en-v1.5` |

## Project Structure

```
rag-dynamic-semantic-window/
├── data/
│   └── source_text.txt          # Test corpus (e.g., Wikipedia article)
├── src/
│   ├── __init__.py
│   ├── dynamic_retriever.py     # DynamicSemanticExpander implementation
│   └── utils.py                 # Data loading, embedding utilities
├── notebook_demo.ipynb          # Interactive demo with visualizations
├── run_benchmark.py             # CLI benchmark script
├── requirements.txt
├── .env.example
├── CONCEPT_DESIGN.md            # Technical design document
└── README.md
```

## Evaluation Metrics

The benchmark evaluates each strategy on:

| Metric | Description |
|--------|-------------|
| **Token Count** | Total tokens consumed by retrieved context |
| **Intra-Cluster Similarity** | Average cosine similarity within expanded chunks (coherence) |
| **Boundary Quality** | Whether context boundaries align with semantic transitions |
| **Answer Relevance** | LLM-judged relevance of the final answer |

Results are exported to `results/benchmark_results.json` and visualized in the notebook.

## License

MIT License — see [LICENSE](LICENSE) for details.
