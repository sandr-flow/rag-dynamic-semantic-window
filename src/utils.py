"""Utility functions for data loading and preprocessing."""

import os
import re
from typing import Optional

from dotenv import load_dotenv
from llama_index.core.schema import TextNode


def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_env(key: str, default: Optional[str] = None) -> str:
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


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.

    Args:
        text: Input text to split.

    Returns:
        List of sentences.
    """
    # Simple sentence splitter - handles common cases
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


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
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
