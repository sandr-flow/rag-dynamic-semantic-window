"""QASPER dataset loader for benchmark testing.

Loads scientific papers from the QASPER dataset (Question Answering on
Scientific Papers) from the Hugging Face hub.

Reads the hub's auto-converted parquet branch directly instead of going
through ``datasets.load_dataset``: QASPER only ships a loading script, which
needs ``datasets==2.21.0``, which in turn pins ``dill<0.3.9`` -- and dill
0.3.8 cannot run on Python 3.14 (it overrides the stdlib pickler's
``_batch_setitems``, whose signature changed). The parquet files carry the
same schema, so nothing downstream changes.
"""

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

QASPER_REPO = "allenai/qasper"
PARQUET_REVISION = "refs/convert/parquet"


def load_qasper_dataset(split: str = "validation") -> list[dict]:
    """
    Load QASPER dataset from Hugging Face.

    Args:
        split: Dataset split ("train", "validation", or "test").

    Returns:
        List of paper dicts.
    """
    path = hf_hub_download(
        QASPER_REPO,
        f"qasper/{split}/0000.parquet",
        repo_type="dataset",
        revision=PARQUET_REVISION,
    )
    return pq.read_table(path).to_pylist()


def extract_full_text(paper: dict) -> str:
    """
    Extract full text from a QASPER paper entry.

    Combines abstract and all section paragraphs into a single text.

    Args:
        paper: A single paper dict from QASPER.

    Returns:
        Full text of the paper as a single string.
    """
    parts = []

    # Add abstract
    abstract = paper.get("abstract")
    if abstract:
        parts.append(str(abstract))

    # Add section contents
    full_text = paper.get("full_text", {})
    if full_text:
        section_names = full_text.get("section_name", []) or []
        paragraphs_list = full_text.get("paragraphs", []) or []

        for section_name, paragraphs in zip(section_names, paragraphs_list, strict=False):
            if section_name:
                parts.append(f"\n{section_name}\n")
            if paragraphs:
                for para in paragraphs:
                    if para and str(para).strip():
                        parts.append(str(para))

    return "\n".join(parts)


def fetch_qasper_articles(
    count: int,
    min_length: int = 2000,
    split: str = "validation",
    skip: int = 0,
) -> list[tuple[str, str]]:
    """
    Fetch articles from QASPER dataset.

    Args:
        count: Number of articles to fetch.
        min_length: Minimum article length in characters.
        split: Dataset split to use.
        skip: Number of valid articles to skip (for non-overlapping sets).

    Returns:
        List of (title, text) tuples.
    """
    papers = load_qasper_dataset(split)
    articles = []
    skipped = 0

    for paper in papers:
        if len(articles) >= count:
            break

        title = paper.get("title", "Untitled")
        text = extract_full_text(paper)

        # Filter by length
        if len(text) < min_length:
            continue

        # Skip first N valid articles
        if skipped < skip:
            skipped += 1
            continue

        articles.append((title, text))
        print(f"  [INFO] [{len(articles)}/{count}] Loaded: {title[:50]}... ({len(text)} chars)")

    return articles
