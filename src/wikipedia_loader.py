"""Wikipedia article loader for benchmark testing."""

import httpx
import wikipediaapi


def fetch_random_article(min_length: int = 2000, max_retries: int = 10) -> tuple[str, str]:
    """
    Fetch a random Wikipedia article with quality filtering.

    Args:
        min_length: Minimum article length in characters.
        max_retries: Maximum attempts to find a suitable article.

    Returns:
        Tuple of (title, text).

    Raises:
        RuntimeError: If no suitable article found after max_retries.
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="DynamicSemanticWindowBenchmark/1.0",
        language="en",
    )

    for attempt in range(max_retries):
        # Use Wikipedia API to get random article title
        response = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "random",
                "rnnamespace": 0,  # Main namespace only (articles)
                "rnlimit": 1,
                "format": "json",
            },
            headers={
                "User-Agent": "DynamicSemanticWindowBenchmark/1.0 (https://github.com/example; contact@example.com)"
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        title = data["query"]["random"][0]["title"]
        page = wiki.page(title)

        if page.exists():
            text = page.text
            if len(text) >= min_length:
                return page.title, text
            else:
                print(f"  Skipping '{title}' ({len(text)} chars < {min_length})")

    raise RuntimeError(
        f"Could not find article with >= {min_length} chars after {max_retries} attempts"
    )


def fetch_article_by_title(title: str) -> tuple[str, str]:
    """
    Fetch a specific Wikipedia article by title.

    Args:
        title: Wikipedia article title.

    Returns:
        Tuple of (title, text).

    Raises:
        ValueError: If article not found.
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="DynamicSemanticWindowBenchmark/1.0",
        language="en",
    )

    page = wiki.page(title)
    if not page.exists():
        raise ValueError(f"Article not found: {title}")

    return page.title, page.text
