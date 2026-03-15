"""Optional bonus: simple word-level bigram Markov model."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any

from .utils import set_random_seed

START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"


def tokenize(text: str, max_vocab_size: int = 1000) -> list[str]:
    """Tokenize text, keep top words, replace rare words with <UNK>, and add boundaries."""
    raw_tokens = re.findall(r"[a-z]+", text.lower())
    if not raw_tokens:
        raise ValueError("No valid word tokens found.")
    counts = Counter(raw_tokens)
    keep_size = max(1, max_vocab_size - 3)
    vocab = {token for token, _ in counts.most_common(keep_size)}
    normalized = [token if token in vocab else UNK_TOKEN for token in raw_tokens]
    return [START_TOKEN] + normalized + [END_TOKEN]


def build_word_model(tokens: list[str]) -> dict[str, Any]:
    """Build a Laplace-smoothed bigram word model."""
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for current_token, next_token in zip(tokens, tokens[1:]):
        counts[current_token][next_token] += 1

    vocab = sorted(set(tokens) | {START_TOKEN, END_TOKEN, UNK_TOKEN})
    probabilities: dict[str, dict[str, float]] = {}
    vocab_size = len(vocab)
    for token in vocab:
        row_counts = counts.get(token, Counter())
        total = sum(row_counts.values())
        probabilities[token] = {
            next_token: (row_counts.get(next_token, 0) + 1) / (total + vocab_size)
            for next_token in vocab
        }

    return {
        "tokens": tokens,
        "vocab": vocab,
        "counts": dict(counts),
        "probabilities": probabilities,
    }


def generate_words(model: dict[str, Any], max_words: int = 50, seed: int | None = None) -> str:
    """Generate a word sequence from the bigram word model."""
    set_random_seed(seed)
    probabilities = model["probabilities"]
    current = START_TOKEN
    generated: list[str] = []
    for _ in range(max_words):
        distribution = probabilities[current]
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        words = [word for word, _ in ranked]
        weights = [weight for _, weight in ranked]
        next_word = random.choices(words, weights=weights, k=1)[0]
        if next_word == END_TOKEN:
            break
        if next_word != UNK_TOKEN:
            generated.append(next_word)
        current = next_word
    return " ".join(generated)
