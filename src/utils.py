"""Utility functions for data loading and preprocessing."""

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from llama_index.core.schema import TextNode


def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_env(key: str, default: str | None = None) -> str:
    """
    Get environment variable value.

    Args:
        key: Environment variable name.
        default: Default value if not found.

    Returns:
        Environment variable value.

    Raises:
        ValueError: If variable not found and no default provided.
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set")
    return value


_PERIOD_PLACEHOLDER = "<prd>"
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "Fig.",
    "fig.",
    "Eq.",
    "eq.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Sr.",
    "Jr.",
    "vs.",
    "et al.",
)


@lru_cache(maxsize=4)
def _pysbd_segmenter(language: str = "en"):
    import pysbd

    return pysbd.Segmenter(language=language, clean=False)


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a shared sentence segmenter.

    Args:
        text: Input text to split.

    Returns:
        List of sentences.
    """
    text = text.strip()
    if not text:
        return []

    try:
        sentences = _pysbd_segmenter().segment(text)
    except ImportError:
        sentences = _fallback_sentence_split(text)
    return [s.strip() for s in sentences if s.strip()]


def _fallback_sentence_split(text: str) -> list[str]:
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(
            abbreviation, abbreviation.replace(".", _PERIOD_PLACEHOLDER)
        )
    protected = re.sub(r"(?<=\d)\.(?=\d)", _PERIOD_PLACEHOLDER, protected)
    protected = re.sub(r"\b([A-Z])\.", rf"\1{_PERIOD_PLACEHOLDER}", protected)

    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return [s.replace(_PERIOD_PLACEHOLDER, ".") for s in sentences]


def build_embedding_texts(sentences: list[str], phantom_window: int = 0) -> list[str]:
    """Build the exact text payload embedded for each sentence."""
    if phantom_window <= 0:
        return list(sentences)

    embedding_texts = []
    for i in range(len(sentences)):
        start_idx = max(0, i - phantom_window)
        end_idx = min(len(sentences), i + phantom_window + 1)
        embedding_texts.append(" ".join(sentences[start_idx:end_idx]))
    return embedding_texts


def create_sentence_nodes(sentences: list[str], doc_id: str = "doc") -> list[TextNode]:
    """
    Create TextNodes from sentences with prev/next linking.

    Args:
        sentences: List of sentence strings.
        doc_id: Document identifier for node IDs.

    Returns:
        List of TextNodes with metadata for neighbor linking.
    """
    nodes = []

    for i, sentence in enumerate(sentences):
        node_id = f"{doc_id}_sent_{i:04d}"
        prev_id = f"{doc_id}_sent_{i - 1:04d}" if i > 0 else None
        next_id = f"{doc_id}_sent_{i + 1:04d}" if i < len(sentences) - 1 else None

        node = TextNode(
            text=sentence,
            id_=node_id,
            metadata={
                "prev_id": prev_id,
                "next_id": next_id,
                "source_doc": doc_id,
                "position": i,
            },
        )
        nodes.append(node)

    return nodes


def load_text_file(filepath: str) -> str:
    """
    Load text content from file.

    Args:
        filepath: Path to text file.

    Returns:
        File contents as string.
    """
    with open(filepath, encoding="utf-8") as f:
        return f.read()
