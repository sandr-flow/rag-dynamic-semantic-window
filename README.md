# Dynamic Semantic Window RAG

Experimental RAG retrieval strategy using dynamic context window expansion based on semantic similarity.

> ⚠️ **Active Experiment**: This project is under active development. Results and configurations may change.

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Benchmark Results](#benchmark-results)
- [License](#license)

## Background

Traditional RAG chunking methods use fixed-size windows, which can split related context or include irrelevant text. This project explores **dynamic semantic window expansion** — starting from a seed sentence and expanding the context window based on cosine similarity with neighboring sentences.

### Key Features

- **Phantom Embeddings**: Embeddings computed using surrounding context for better semantic representation
- **Two-Pass Retrieval**: Broad initial search (top_k × multiplier), then refined expansion
- **Adaptive Thresholds**: Dynamic expansion based on local density and gradient detection
- **Query-Aware Expansion**: Considers both neighbor similarity and query relevance

## Installation

### Prerequisites

- Python 3.10+
- Mistral API key (for QA generation)

### Steps

```bash
# Clone repository
git clone https://github.com/your-username/rag-dynamic-semantic-window.git
cd rag-dynamic-semantic-window

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Benchmark

```bash
# Wikipedia articles
python run_benchmark.py --source wikipedia --num-articles 30 --num-questions 3 --min-length 6000

# QASPER scientific papers
python run_benchmark.py --source qasper --num-articles 30 --num-questions 3 --min-length 4000
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Data source: `static`, `wikipedia`, `qasper` | `static` |
| `--num-articles` | Number of articles to benchmark | `5` |
| `--num-questions` | Questions per article | `3` |
| `--min-length` | Minimum article length (chars) | `2000` |

## Configuration

Create `.env` file based on `.env.example`:

| Variable | Description | Required |
|----------|-------------|----------|
| `MISTRAL_API_KEY` | Mistral AI API key for QA generation | Yes |
| `EMBEDDING_MODEL` | HuggingFace embedding model | No |
| `SIMILARITY_THRESHOLD` | Expansion similarity threshold | No |
| `MAX_EXPAND` | Maximum expansion per direction | No |

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
