"""Central configuration for the Markov text generation project."""

from __future__ import annotations

from pathlib import Path

DEFAULT_TRAIN_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
DEFAULT_TEST_SAME_URL = "https://www.gutenberg.org/cache/epub/12/pg12.txt"
DEFAULT_TEST_DIFF_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"

MAX_FETCH_CHARS = 50_000

VOCAB = ["^", "$", " "] + list("abcdefghijklmnopqrstuvwxyz")
VOCAB_SIZE = len(VOCAB)

DEFAULT_OUTPUT_DIRS = {
    "raw": Path("data/raw"),
    "processed": Path("data/processed"),
    "samples": Path("outputs/samples"),
    "metrics": Path("outputs/metrics"),
    "figures": Path("outputs/figures"),
}

DEFAULT_RANDOM_SEED = 42
DEFAULT_TOP_K = 5
DEFAULT_MAX_GEN_LENGTH = 200
DEFAULT_ORDERS = [1, 3, 5]
