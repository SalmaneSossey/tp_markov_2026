"""Higher-order character Markov models."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .config import DEFAULT_MAX_GEN_LENGTH, VOCAB, VOCAB_SIZE
from .part2_preprocessing import preprocess_for_order
from .part4_scoring import compute_perplexity
from .part5_generation import sample_from_distribution
from .utils import clean_generated_text, set_random_seed


def preprocess_for_order_n(text: str, order: int) -> str:
    """Preprocess text with repeated start markers for order-n modeling."""
    return preprocess_for_order(text, order)


def _uniform_distribution() -> dict[str, float]:
    return {symbol: 1.0 / VOCAB_SIZE for symbol in VOCAB}


def build_high_order_model(text: str, order: int = 3) -> dict[str, Any]:
    """Build an arbitrary order-n character model using tuple histories."""
    if order < 1:
        raise ValueError("Order must be at least 1.")

    processed = preprocess_for_order_n(text, order)
    counts: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for index in range(order, len(processed)):
        history = tuple(processed[index - order:index])
        next_char = processed[index]
        counts[history][next_char] += 1

    probabilities: dict[tuple[str, ...], dict[str, float]] = {}
    for history, history_counts in counts.items():
        total = sum(history_counts.values())
        probabilities[history] = {
            symbol: (history_counts.get(symbol, 0) + 1) / (total + VOCAB_SIZE)
            for symbol in VOCAB
        }

    possible_states = VOCAB_SIZE**order
    state_count = len(probabilities)
    sparsity = state_count / possible_states if possible_states else 0.0

    return {
        "order": order,
        "vocab": VOCAB,
        "vocab_size": VOCAB_SIZE,
        "processed_text": processed,
        "counts": dict(counts),
        "probabilities": probabilities,
        "fallback_distribution": _uniform_distribution(),
        "state_count": state_count,
        "possible_states": possible_states,
        "sparsity": sparsity,
    }


def compute_log_likelihood_order_n(model: dict[str, Any], text: str, order: int) -> float:
    """Compute log-likelihood for an order-n model."""
    processed = preprocess_for_order_n(text, order)
    probabilities = model["probabilities"]
    fallback_distribution = model.get("fallback_distribution", _uniform_distribution())
    log_likelihood = 0.0
    for index in range(order, len(processed)):
        history = tuple(processed[index - order:index])
        next_char = processed[index]
        distribution = probabilities.get(history, fallback_distribution)
        probability = distribution.get(next_char, fallback_distribution[next_char])
        log_likelihood += math.log(probability)
    return log_likelihood


def generate_high_order(
    model: dict[str, Any],
    order: int = 3,
    max_length: int = DEFAULT_MAX_GEN_LENGTH,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Generate text from an order-n model."""
    set_random_seed(seed)
    probabilities = model["probabilities"]
    fallback_distribution = model.get("fallback_distribution", _uniform_distribution())
    history = ["^"] * order
    generated_chars: list[str] = []
    for _ in range(max_length):
        history_key = tuple(history[-order:])
        distribution = probabilities.get(history_key, fallback_distribution)
        next_char = sample_from_distribution(distribution, top_k=top_k)
        if next_char == "$":
            return ("^" * order) + "".join(generated_chars) + "$"
        generated_chars.append(next_char)
        history.append(next_char)
    return ("^" * order) + "".join(generated_chars)


def _evaluate_order_split(model: dict[str, Any], raw_text: str, order: int) -> dict[str, float | int]:
    processed = preprocess_for_order_n(raw_text, order)
    n_transitions = len(processed) - order
    log_likelihood = compute_log_likelihood_order_n(model, raw_text, order)
    avg_log_likelihood = log_likelihood / n_transitions
    perplexity = compute_perplexity(log_likelihood, n_transitions)
    return {
        "n_transitions": n_transitions,
        "log_likelihood": log_likelihood,
        "avg_log_likelihood": avg_log_likelihood,
        "perplexity": perplexity,
    }


def compare_orders(train_text: str, test_same: str, test_diff: str, orders: list[int]) -> dict[str, Any]:
    """Train and evaluate multiple character model orders."""
    results: dict[str, Any] = {}
    for order in orders:
        model = build_high_order_model(train_text, order=order)
        sample = generate_high_order(model, order=order, top_k=5, seed=42)
        results[str(order)] = {
            "order": order,
            "state_count": model["state_count"],
            "possible_states": model["possible_states"],
            "sparsity": model["sparsity"],
            "metrics": {
                "train": _evaluate_order_split(model, train_text, order),
                "same_style": _evaluate_order_split(model, test_same, order),
                "different_style": _evaluate_order_split(model, test_diff, order),
            },
            "sample": {
                "raw": sample,
                "cleaned": clean_generated_text(sample),
            },
        }
    return results
