"""First-order character-level Markov model."""

from __future__ import annotations

from collections import Counter, defaultdict

from .config import VOCAB, VOCAB_SIZE
from .part2_preprocessing import preprocess


def count_transitions(text: str) -> dict[str, Counter[str]]:
    """Count adjacent character transitions over preprocessed text."""
    processed = text if text.startswith("^") and text.endswith("$") else preprocess(text)
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for current_char, next_char in zip(processed, processed[1:]):
        counts[current_char][next_char] += 1
    for symbol in VOCAB:
        counts.setdefault(symbol, Counter())
    return dict(counts)


def build_probability_matrix(counts: dict[str, Counter[str]], vocab: list[str]) -> dict[str, dict[str, float]]:
    """Apply Laplace smoothing to produce a full transition matrix."""
    model: dict[str, dict[str, float]] = {}
    vocab_size = len(vocab)
    for current_char in vocab:
        row_counts = counts.get(current_char, Counter())
        total = sum(row_counts.values())
        model[current_char] = {}
        for next_char in vocab:
            model[current_char][next_char] = (row_counts.get(next_char, 0) + 1) / (total + vocab_size)
    return model


def verify_model(model: dict[str, dict[str, float]], vocab: list[str]) -> bool:
    """Check that the model has all rows and each row sums to 1."""
    for current_char in vocab:
        if current_char not in model:
            return False
        row = model[current_char]
        if any(symbol not in row for symbol in vocab):
            return False
        if abs(sum(row.values()) - 1.0) > 1e-9:
            return False
    return True


def train_order1_model(raw_text: str) -> dict[str, object]:
    """Train a first-order model and package counts, probabilities, and metadata."""
    processed = preprocess(raw_text)
    counts = count_transitions(processed)
    probabilities = build_probability_matrix(counts, VOCAB)
    if not verify_model(probabilities, VOCAB):
        raise ValueError("Order-1 model verification failed.")
    return {
        "order": 1,
        "vocab": VOCAB,
        "vocab_size": VOCAB_SIZE,
        "processed_text": processed,
        "counts": counts,
        "probabilities": probabilities,
    }


def get_top_transitions(counts: dict[str, Counter[str]], top_n: int = 10) -> list[dict[str, object]]:
    """Return the most frequent first-order transitions."""
    flattened: list[tuple[str, str, int]] = []
    for current_char in sorted(counts):
        for next_char, count in counts[current_char].items():
            flattened.append((current_char, next_char, count))
    flattened.sort(key=lambda item: (-item[2], item[0], item[1]))
    return [
        {"from": current_char, "to": next_char, "count": count}
        for current_char, next_char, count in flattened[:top_n]
    ]
