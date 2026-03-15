"""Utility helpers for file IO, formatting, and reproducibility."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_text(path: str | Path, text: str) -> Path:
    """Save a UTF-8 text file, creating parent directories if needed."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(text, encoding="utf-8")
    return path_obj


def save_json(path: str | Path, obj: Any) -> Path:
    """Save JSON with stable formatting for later report use."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path_obj


def pretty_print_header(title: str) -> None:
    """Print a compact console section header."""
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def clean_generated_text(text: str) -> str:
    """Remove training markers and normalize spaces for display."""
    cleaned = text.replace("^", "").replace("$", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def set_random_seed(seed: int | None) -> None:
    """Seed the Python RNG when reproducibility is requested."""
    if seed is not None:
        random.seed(seed)


def top_n_items(items: dict[Any, Any] | Iterable[tuple[Any, Any]], top_n: int = 10) -> list[tuple[Any, Any]]:
    """Return the top-N items sorted by descending value then ascending key."""
    if isinstance(items, dict):
        pairs = list(items.items())
    else:
        pairs = list(items)
    return sorted(pairs, key=lambda item: (-item[1], str(item[0])))[:top_n]
