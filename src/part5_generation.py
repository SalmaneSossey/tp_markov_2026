"""Sampling and generation for first-order character models."""

from __future__ import annotations

import random
from typing import Any

from .config import DEFAULT_MAX_GEN_LENGTH, DEFAULT_RANDOM_SEED, DEFAULT_TOP_K, VOCAB
from .utils import clean_generated_text, set_random_seed


def _extract_probability_matrix(model: dict[str, Any]) -> dict[str, dict[str, float]]:
    if "probabilities" in model:
        return model["probabilities"]
    return model


def sample_from_distribution(prob_dist: dict[str, float], top_k: int | None = None) -> str:
    """Sample a symbol from a probability distribution with optional top-k restriction."""
    ranked = sorted(prob_dist.items(), key=lambda item: (-item[1], item[0]))
    if top_k == 1:
        return ranked[0][0]
    if top_k is not None and top_k > 1:
        ranked = ranked[:top_k]
    symbols = [symbol for symbol, _ in ranked]
    weights = [weight for _, weight in ranked]
    total = sum(weights)
    normalized = [weight / total for weight in weights]
    return random.choices(symbols, weights=normalized, k=1)[0]


def generate_text(
    model: dict[str, Any],
    max_length: int = DEFAULT_MAX_GEN_LENGTH,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Generate text from a first-order model."""
    set_random_seed(seed)
    probability_matrix = _extract_probability_matrix(model)
    generated_chars: list[str] = []
    current_char = "^"
    for _ in range(max_length):
        distribution = probability_matrix.get(
            current_char,
            {symbol: 1.0 / len(VOCAB) for symbol in VOCAB},
        )
        next_char = sample_from_distribution(distribution, top_k=top_k)
        if next_char == "$":
            return "^" + "".join(generated_chars) + "$"
        generated_chars.append(next_char)
        current_char = next_char
    return "^" + "".join(generated_chars)


def compare_sampling_strategies(model: dict[str, Any], num_samples: int = 3) -> dict[str, list[dict[str, str | int | None]]]:
    """Generate multiple samples using full sampling, top-k, and greedy decoding."""
    strategies = {
        "full_sampling": None,
        "top_k_sampling": DEFAULT_TOP_K,
        "greedy": 1,
    }
    results: dict[str, list[dict[str, str | int | None]]] = {}
    for strategy_name, top_k in strategies.items():
        samples = []
        for sample_index in range(num_samples):
            seed = DEFAULT_RANDOM_SEED + sample_index
            raw = generate_text(model, top_k=top_k, seed=seed)
            samples.append(
                {
                    "seed": seed,
                    "top_k": top_k,
                    "raw": raw,
                    "cleaned": clean_generated_text(raw),
                }
            )
        results[strategy_name] = samples
    return results
