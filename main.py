"""End-to-end experiment pipeline for TP_Markov2026."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import (
    DEFAULT_MAX_GEN_LENGTH,
    DEFAULT_ORDERS,
    DEFAULT_OUTPUT_DIRS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_DIFF_URL,
    DEFAULT_TEST_SAME_URL,
    DEFAULT_TOP_K,
    DEFAULT_TRAIN_URL,
    MAX_FETCH_CHARS,
)
from src.part1_scraping import save_raw_text
from src.part2_preprocessing import preprocess
from src.part3_order1_model import get_top_transitions, train_order1_model
from src.part4_scoring import evaluate_model
from src.part5_generation import compare_sampling_strategies
from src.part6_orderN_model import build_high_order_model, compare_orders, generate_high_order
from src.part7_wordlevel import build_word_model, generate_words, tokenize
from src.utils import clean_generated_text, ensure_dir, pretty_print_header, save_json, save_text

FALLBACK_TEXTS = {
    "train": (
        "Alice was beginning to get very tired of sitting by her sister on the bank "
        "and of having nothing to do once or twice she had peeped into the book her "
        "sister was reading but it had no pictures or conversations in it and what "
        "is the use of a book thought Alice without pictures or conversation "
        "so she was considering in her own mind as well as she could for the hot day "
        "made her feel very sleepy and stupid whether the pleasure of making a daisy "
        "chain would be worth the trouble of getting up and picking the daisies"
    ),
    "same": (
        "One thing was certain that the white kitten had had nothing to do with it "
        "it was the black kitten fault entirely for the white kitten had been having "
        "its face washed by the old cat for the last quarter of an hour and bearing "
        "it pretty well considering so you see that it could have had no hand in the "
        "mischief"
    ),
    "diff": (
        "It is a truth universally acknowledged that a single man in possession of a "
        "good fortune must be in want of a wife however little known the feelings or "
        "views of such a man may be on his first entering a neighbourhood this truth "
        "is so well fixed in the minds of the surrounding families"
    ),
}


def _read_cached_text(path: Path) -> str | None:
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    return None


def _load_or_fetch_text(url: str, cache_path: Path, fallback_text: str) -> tuple[str, dict[str, str]]:
    cached = _read_cached_text(cache_path)
    if cached is not None:
        return cached, {"mode": "cache", "url": url, "path": str(cache_path)}
    try:
        text = save_raw_text(url, cache_path, max_chars=MAX_FETCH_CHARS)
        return text, {"mode": "fetched", "url": url, "path": str(cache_path)}
    except Exception as exc:
        save_text(cache_path, fallback_text)
        return fallback_text, {
            "mode": "fallback",
            "url": url,
            "path": str(cache_path),
            "reason": str(exc),
        }


def _save_processed_texts(processed_texts: dict[str, str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, text in processed_texts.items():
        path = DEFAULT_OUTPUT_DIRS["processed"] / f"{name}_processed.txt"
        save_text(path, text)
        paths[name] = str(path)
    return paths


def _write_samples_file(path: Path, title: str, samples: list[dict[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    for index, sample in enumerate(samples, start=1):
        lines.append(f"## Sample {index}")
        for key, value in sample.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    text = "\n".join(lines).strip() + "\n"
    save_text(path, text)
    return text


def _build_report_notes(summary: dict[str, Any]) -> str:
    top_transitions = summary["order1"]["top_transitions"]
    order1_metrics = summary["order1"]["metrics"]
    order_compare = summary["order_comparison"]
    sources = summary["data_sources"]
    sample_o1 = summary["order1"]["samples"]["top_k_sampling"][0]["cleaned"]
    sample_o3 = summary["order3"]["samples"][0]["cleaned"]

    lines = [
        "# Report Notes",
        "",
        "## Data Source",
        f"- Train URL: {sources['train']['url']} ({sources['train']['mode']})",
        f"- Same-style URL: {sources['same_style']['url']} ({sources['same_style']['mode']})",
        f"- Different-style URL: {sources['different_style']['url']} ({sources['different_style']['mode']})",
        "",
        "## Preprocessing Choices",
        "- Lowercased all text.",
        "- Kept only letters a-z and spaces.",
        "- Collapsed repeated whitespace to one space.",
        "- Added `^` start marker and `$` end marker.",
        "",
        "## Top Transitions",
    ]
    lines.extend(
        [f"- {item['from']} -> {item['to']}: {item['count']}" for item in top_transitions]
    )
    lines.extend(
        [
            "",
            "## Generation Examples",
            f"- Order-1 top-k sample: {sample_o1}",
            f"- Order-3 top-k sample: {sample_o3}",
            "",
            "## Perplexity Comparison",
            f"- Order-1 train perplexity: {order1_metrics['train']['perplexity']:.4f}",
            f"- Order-1 same-style perplexity: {order1_metrics['same_style']['perplexity']:.4f}",
            f"- Order-1 different-style perplexity: {order1_metrics['different_style']['perplexity']:.4f}",
        ]
    )
    for order_key in sorted(order_compare, key=int):
        lines.append(
            f"- Order-{order_key} same-style perplexity: "
            f"{order_compare[order_key]['metrics']['same_style']['perplexity']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Order-1 vs Order-3 Discussion",
            "- Higher-order models keep longer local patterns but become much sparser.",
            (
                f"- Order-3 used {order_compare['3']['state_count']} states out of "
                f"{order_compare['3']['possible_states']} possible histories."
            ),
            "",
            "## Relation to LLMs",
            "- Markov chains only condition on a short fixed history, unlike modern LLMs with much longer context windows.",
            "- Perplexity is still useful as a shared evaluation idea, even though the models are much simpler here.",
            "",
            "## Limitations",
            "- Restricted vocabulary removes punctuation and capitalization.",
            "- Character-level outputs are locally plausible but often globally incoherent.",
            "- Higher orders improve short patterns but increase sparsity quickly.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_pipeline(run_word_level: bool = True) -> dict[str, Any]:
    """Run the full experiment pipeline and save outputs."""
    for path in DEFAULT_OUTPUT_DIRS.values():
        ensure_dir(path)

    pretty_print_header("Fetching or Loading Data")
    train_text, train_source = _load_or_fetch_text(
        DEFAULT_TRAIN_URL,
        DEFAULT_OUTPUT_DIRS["raw"] / "train.txt",
        FALLBACK_TEXTS["train"],
    )
    test_same_text, same_source = _load_or_fetch_text(
        DEFAULT_TEST_SAME_URL,
        DEFAULT_OUTPUT_DIRS["raw"] / "test_same.txt",
        FALLBACK_TEXTS["same"],
    )
    test_diff_text, diff_source = _load_or_fetch_text(
        DEFAULT_TEST_DIFF_URL,
        DEFAULT_OUTPUT_DIRS["raw"] / "test_diff.txt",
        FALLBACK_TEXTS["diff"],
    )
    print(f"Train text source: {train_source['mode']}")
    print(f"Same-style text source: {same_source['mode']}")
    print(f"Different-style text source: {diff_source['mode']}")

    pretty_print_header("Preprocessing")
    processed_texts = {
        "train": preprocess(train_text),
        "test_same": preprocess(test_same_text),
        "test_diff": preprocess(test_diff_text),
    }
    processed_paths = _save_processed_texts(processed_texts)
    for name, path in processed_paths.items():
        print(f"Saved processed {name}: {path}")

    pretty_print_header("Training Order-1 Model")
    order1_model = train_order1_model(train_text)
    top_transitions = get_top_transitions(order1_model["counts"], top_n=10)
    order1_metrics = evaluate_model(order1_model, train_text, test_same_text, test_diff_text)
    order1_samples = compare_sampling_strategies(order1_model, num_samples=3)

    pretty_print_header("Training Higher-Order Models")
    order_comparison = compare_orders(train_text, test_same_text, test_diff_text, DEFAULT_ORDERS)
    order3_model = build_high_order_model(train_text, order=3)
    order3_samples = []
    for sample_index in range(3):
        seed = DEFAULT_RANDOM_SEED + sample_index
        raw_sample = generate_high_order(
            order3_model,
            order=3,
            max_length=DEFAULT_MAX_GEN_LENGTH,
            top_k=DEFAULT_TOP_K,
            seed=seed,
        )
        order3_samples.append(
            {
                "seed": seed,
                "top_k": DEFAULT_TOP_K,
                "raw": raw_sample,
                "cleaned": clean_generated_text(raw_sample),
            }
        )

    order5_samples = []
    if "5" in order_comparison:
        order5_model = build_high_order_model(train_text, order=5)
        for sample_index in range(3):
            seed = DEFAULT_RANDOM_SEED + sample_index
            raw_sample = generate_high_order(
                order5_model,
                order=5,
                max_length=DEFAULT_MAX_GEN_LENGTH,
                top_k=DEFAULT_TOP_K,
                seed=seed,
            )
            order5_samples.append(
                {
                    "seed": seed,
                    "top_k": DEFAULT_TOP_K,
                    "raw": raw_sample,
                    "cleaned": clean_generated_text(raw_sample),
                }
            )

    pretty_print_header("Optional Word-Level Model")
    word_level_summary: dict[str, Any] = {"enabled": run_word_level, "samples": []}
    if run_word_level:
        word_tokens = tokenize(train_text, max_vocab_size=1000)
        word_model = build_word_model(word_tokens)
        word_level_summary["vocab_size"] = len(word_model["vocab"])
        word_level_summary["samples"] = [
            generate_words(word_model, max_words=50, seed=DEFAULT_RANDOM_SEED + sample_index)
            for sample_index in range(3)
        ]
        print(f"Word-level vocab size: {word_level_summary['vocab_size']}")
    else:
        print("Word-level model skipped.")

    pretty_print_header("Saving Artifacts")
    save_json(DEFAULT_OUTPUT_DIRS["metrics"] / "order1_metrics.json", order1_metrics)
    save_json(DEFAULT_OUTPUT_DIRS["metrics"] / "order_comparison.json", order_comparison)
    save_json(DEFAULT_OUTPUT_DIRS["metrics"] / "top_transitions_order1.json", top_transitions)

    _write_samples_file(
        DEFAULT_OUTPUT_DIRS["samples"] / "order1_samples.md",
        "Order-1 Samples",
        order1_samples["full_sampling"]
        + order1_samples["top_k_sampling"]
        + order1_samples["greedy"],
    )
    _write_samples_file(
        DEFAULT_OUTPUT_DIRS["samples"] / "order3_topk_samples.md",
        "Order-3 Top-k Samples",
        order3_samples,
    )
    if order5_samples:
        _write_samples_file(
            DEFAULT_OUTPUT_DIRS["samples"] / "order5_topk_samples.md",
            "Order-5 Top-k Samples",
            order5_samples,
        )
    if word_level_summary["samples"]:
        save_text(
            DEFAULT_OUTPUT_DIRS["samples"] / "wordlevel_samples.txt",
            "\n".join(word_level_summary["samples"]) + "\n",
        )

    summary = {
        "data_sources": {
            "train": train_source,
            "same_style": same_source,
            "different_style": diff_source,
        },
        "processed_paths": processed_paths,
        "order1": {
            "metrics": order1_metrics,
            "top_transitions": top_transitions,
            "samples": order1_samples,
        },
        "order3": {
            "metrics": order_comparison["3"]["metrics"],
            "samples": order3_samples,
        },
        "order5": {
            "metrics": order_comparison.get("5", {}).get("metrics", {}),
            "samples": order5_samples,
        },
        "order_comparison": order_comparison,
        "word_level": word_level_summary,
    }
    save_json(DEFAULT_OUTPUT_DIRS["metrics"] / "summary.json", summary)

    report_notes = _build_report_notes(summary)
    save_text(Path("report_notes.md"), report_notes)
    save_text(DEFAULT_OUTPUT_DIRS["metrics"] / "report_summary.md", report_notes)

    pretty_print_header("Summary")
    print(f"Order-1 same-style perplexity: {order1_metrics['same_style']['perplexity']:.4f}")
    print(f"Order-3 same-style perplexity: {order_comparison['3']['metrics']['same_style']['perplexity']:.4f}")
    if "5" in order_comparison:
        print(f"Order-5 same-style perplexity: {order_comparison['5']['metrics']['same_style']['perplexity']:.4f}")
    print(f"Saved metrics to: {DEFAULT_OUTPUT_DIRS['metrics']}")
    print(f"Saved samples to: {DEFAULT_OUTPUT_DIRS['samples']}")

    return summary


if __name__ == "__main__":
    run_pipeline()
