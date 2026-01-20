"""Wikipedia article loader for benchmark testing with parallel fetching."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
import wikipediaapi


# Reusable Wikipedia client (created once)
_wiki_client: Optional[wikipediaapi.Wikipedia] = None


def _get_wiki_client() -> wikipediaapi.Wikipedia:
    """Get or create singleton Wikipedia client."""
    global _wiki_client
    if _wiki_client is None:
        _wiki_client = wikipediaapi.Wikipedia(
            user_agent="DynamicSemanticWindowBenchmark/1.0",
            language="en",
        )
    return _wiki_client


def _fetch_random_titles(count: int = 10) -> list[str]:
    """
    Fetch multiple random article titles in a single API call.

    Args:
        count: Number of random titles to fetch (max 500).

    Returns:
        List of article titles.
    """
    try:
        response = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "random",
                "rnnamespace": 0,  # Main namespace only
                "rnlimit": min(count, 500),
                "format": "json",
            },
            headers={
                "User-Agent": "DynamicSemanticWindowBenchmark/1.0"
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return [item["title"] for item in data["query"]["random"]]
    except Exception as e:
        print(f"  ⚠️ Error fetching random titles: {e}")
        return []


def _fetch_single_article(title: str, min_length: int = 2000) -> Optional[tuple[str, str]]:
    """
    Fetch a single article by title (for use in thread pool).

    Args:
        title: Article title.
        min_length: Minimum acceptable length.

    Returns:
        Tuple of (title, text) or None if too short/not found.
    """
    try:
        wiki = _get_wiki_client()
        page = wiki.page(title)
        if page.exists() and len(page.text) >= min_length:
            return page.title, page.text
    except Exception as e:
        print(f"  ⚠️ Error fetching '{title}': {e}")
    return None


async def fetch_random_articles_batch(
    count: int,
    min_length: int = 2000,
    max_workers: int = 5,
    batch_size: int = 10,
    batch_delay: float = 1.0,
) -> list[tuple[str, str]]:
    """
    Fetch multiple random Wikipedia articles in parallel batches.

    Uses batch title fetching + parallel content retrieval for speed.
    Processes in smaller batches with delays to avoid rate limits.

    Args:
        count: Number of articles to fetch.
        min_length: Minimum article length in characters.
        max_workers: Number of parallel fetch threads.
        batch_size: Number of articles to fetch per batch.
        batch_delay: Seconds to wait between batches.

    Returns:
        List of (title, text) tuples.
    """
    articles = []
    attempts = 0
    max_attempts = count * 5  # Allow retries for short articles

    while len(articles) < count and attempts < max_attempts:
        # Fetch batch of random titles
        needed = min(count - len(articles), batch_size)
        titles = _fetch_random_titles(min(needed * 2, 20))

        if not titles:
            attempts += 1
            await asyncio.sleep(batch_delay)
            continue

        # Parallel fetch article content (limited batch)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, _fetch_single_article, title, min_length)
                for title in titles[:batch_size]
            ]
            results = await asyncio.gather(*futures)

        # Collect valid articles
        for result in results:
            if result and len(articles) < count:
                articles.append(result)
                print(f"  📄 [{len(articles)}/{count}] Fetched: {result[0]} ({len(result[1])} chars)")

        attempts += 1
        
        # Delay between batches to avoid rate limits
        if len(articles) < count:
            await asyncio.sleep(batch_delay)

    return articles


def fetch_random_article(min_length: int = 2000, max_retries: int = None) -> tuple[str, str]:
    """
    Fetch a random Wikipedia article with quality filtering.

    Legacy synchronous interface for backward compatibility.

    Args:
        min_length: Minimum article length in characters.
        max_retries: Ignored (loops indefinitely).

    Returns:
        Tuple of (title, text).
    """
    wiki = _get_wiki_client()

    while True:
        titles = _fetch_random_titles(5)
        for title in titles:
            result = _fetch_single_article(title, min_length)
            if result:
                return result


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
    wiki = _get_wiki_client()
    page = wiki.page(title)
    if not page.exists():
        raise ValueError(f"Article not found: {title}")
    return page.title, page.text
