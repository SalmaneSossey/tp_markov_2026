"""Model scoring utilities for log-likelihood and perplexity."""

from __future__ import annotations

import math
from typing import Any

from .config import VOCAB_SIZE
from .part2_preprocessing import preprocess

SAFE_FALLBACK_PROB = 1.0 / VOCAB_SIZE


def _extract_probability_matrix(model: dict[str, Any]) -> dict[str, dict[str, float]]:
    if "probabilities" in model:
        return model["probabilities"]
    return model


def compute_log_likelihood(model: dict[str, Any], test_text: str) -> float:
    """Compute the log-likelihood of preprocessed test text under an order-1 model."""
    processed = preprocess(test_text)
    probability_matrix = _extract_probability_matrix(model)
    log_likelihood = 0.0
    for current_char, next_char in zip(processed, processed[1:]):
        row = probability_matrix.get(current_char)
        probability = row.get(next_char) if row else None
        probability = probability if probability and probability > 0 else SAFE_FALLBACK_PROB
        log_likelihood += math.log(probability)
    return log_likelihood


def compute_perplexity(log_likelihood: float, n_transitions: int) -> float:
    """Compute perplexity from a log-likelihood and transition count."""
    if n_transitions <= 0:
        raise ValueError("n_transitions must be positive.")
    return math.exp(-log_likelihood / n_transitions)


def _evaluate_split(model: dict[str, Any], raw_text: str) -> dict[str, float | int]:
    processed = preprocess(raw_text)
    n_transitions = len(processed) - 1
    log_likelihood = compute_log_likelihood(model, raw_text)
    avg_log_likelihood = log_likelihood / n_transitions
    perplexity = compute_perplexity(log_likelihood, n_transitions)
    return {
        "n_transitions": n_transitions,
        "log_likelihood": log_likelihood,
        "avg_log_likelihood": avg_log_likelihood,
        "perplexity": perplexity,
    }


def evaluate_model(model: dict[str, Any], train_text: str, test_same: str, test_diff: str) -> dict[str, dict[str, float | int]]:
    """Evaluate a first-order model on train, same-style, and different-style text."""
    return {
        "train": _evaluate_split(model, train_text),
        "same_style": _evaluate_split(model, test_same),
        "different_style": _evaluate_split(model, test_diff),
    }
