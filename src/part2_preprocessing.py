"""Text preprocessing for the restricted character vocabulary."""

from __future__ import annotations

import re

from .config import VOCAB, VOCAB_SIZE


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z\s]+", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("Text contains no valid lowercase letters or spaces after preprocessing.")
    return cleaned


def preprocess(text: str) -> str:
    """Preprocess raw text into '^text$' format."""
    return f"^{_clean_text(text)}$"


def preprocess_for_order(text: str, order: int) -> str:
    """Preprocess raw text using repeated start markers for an order-n model."""
    if order < 1:
        raise ValueError("Order must be at least 1.")
    return f"{'^' * order}{_clean_text(text)}$"


def main() -> None:
    """Run a few basic preprocessing checks."""
    tests = [
        "Hello, WORLD!!!",
        "123 ... ???",
        "A   b\tc\n\nD",
    ]
    for sample in tests:
        try:
            print(sample, "->", preprocess(sample))
        except ValueError as exc:
            print(sample, "->", exc)
    print("VOCAB_SIZE =", VOCAB_SIZE)
    print("VOCAB =", VOCAB)


if __name__ == "__main__":
    main()
