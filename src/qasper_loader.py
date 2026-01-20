"""QASPER dataset loader for benchmark testing.

Loads scientific papers from the QASPER dataset (Question Answering on
Scientific Papers) via Hugging Face datasets library.

Requires: datasets==2.21.0 (newer versions removed script support)
Install: pip install datasets==2.21.0
"""

from datasets import load_dataset


def load_qasper_dataset(split: str = "validation") -> list[dict]:
    """
    Load QASPER dataset from Hugging Face.

    Args:
        split: Dataset split ("train", "validation", or "test").

    Returns:
        List of paper dicts.
    """
    dataset = load_dataset("allenai/qasper", split=split, trust_remote_code=True)
    return list(dataset)


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

        for section_name, paragraphs in zip(section_names, paragraphs_list):
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
) -> list[tuple[str, str]]:
    """
    Fetch articles from QASPER dataset.

    Args:
        count: Number of articles to fetch.
        min_length: Minimum article length in characters.
        split: Dataset split to use.

    Returns:
        List of (title, text) tuples.
    """
    papers = load_qasper_dataset(split)
    articles = []

    for paper in papers:
        if len(articles) >= count:
            break

        title = paper.get("title", "Untitled")
        text = extract_full_text(paper)

        # Filter by length
        if len(text) < min_length:
            continue

        articles.append((title, text))
        print(f"  📄 [{len(articles)}/{count}] Loaded: {title[:50]}... ({len(text)} chars)")

    return articles
