"""Fetch raw text data from public URLs."""

from __future__ import annotations

from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import DEFAULT_TRAIN_URL, MAX_FETCH_CHARS
from .utils import save_text

USER_AGENT = "TP_Markov2026/1.0 (+https://www.gutenberg.org)"


def _looks_like_html(content_type: str, text: str) -> bool:
    header = (content_type or "").lower()
    snippet = text[:500].lower().lstrip()
    return "html" in header or snippet.startswith("<!doctype html") or snippet.startswith("<html")


def fetch_text(url: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """Fetch plain text or extract paragraph text from HTML."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch '{url}': {exc}") from exc

    if _looks_like_html(response.headers.get("Content-Type", ""), response.text):
        soup = BeautifulSoup(response.text, "lxml")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(paragraphs).strip()
    else:
        text = response.text.strip()

    if not text:
        raise ValueError(f"No text content found at '{url}'.")

    return text[:max_chars].strip()


def save_raw_text(url: str, destination: str | Path, max_chars: int = MAX_FETCH_CHARS) -> str:
    """Fetch and cache raw text to disk."""
    text = fetch_text(url=url, max_chars=max_chars)
    save_text(destination, text)
    return text


def main() -> None:
    """Simple manual test for text fetching."""
    preview = fetch_text(DEFAULT_TRAIN_URL, max_chars=1000)
    print(preview[:500])


if __name__ == "__main__":
    main()
