"""Convenience entry point for running all experiments from the project root."""

from __future__ import annotations

from main import run_pipeline


def main() -> None:
    summary = run_pipeline(run_word_level=True)
    print("\nRun complete.")
    print("Available summary sections:", ", ".join(sorted(summary.keys())))


if __name__ == "__main__":
    main()
