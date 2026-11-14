"""Command line interface: ``srt demo --reviews reviews.json --game "Game Name"``."""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from steam_review_translator import __version__
from steam_review_translator.core.config import load_dotenv, load_pipeline_config
from steam_review_translator.core.schemas import (
    PipelineInput,
    PipelineOutput,
    Polarity,
    reviews_from_json_file,
)


def format_report(output: PipelineOutput, n_reviews: int | None = None) -> str:
    """Render a human-readable report of the three task outputs."""
    header = f"=== Steam Review Translator: {output.game}"
    header += f" ({n_reviews} reviews) ===" if n_reviews is not None else " ==="
    lines = [header, "", "[Task 1] Sentiment distribution"]
    for polarity, share in output.sentiment.as_dict().items():
        lines.append(f"  {polarity.value:<10} {share:6.1%}")
    lines += [f"  dominant:  {output.sentiment.dominant.value}", "", "[Task 2] Aspect table"]
    if output.aspects:
        lines.append(f"  {'aspect':<12} | {'freq':>4} | polarity")
        lines.append(f"  {'-' * 12}-+-{'-' * 4}-+-{'-' * len(Polarity.POSITIVE)}")
        for e in output.aspects:
            lines.append(f"  {e.aspect.value:<12} | {e.freq:>4} | {e.polarity.value}")
    else:
        lines.append("  (no aspects found)")
    lines += ["", "[Task 3] Meta-review"]
    lines += textwrap.wrap(
        output.meta_review, width=88, initial_indent="  ", subsequent_indent="  "
    )
    return "\n".join(lines)


def _cmd_demo(args: argparse.Namespace) -> int:
    from steam_review_translator.pipeline import build_pipeline  # imports logging setup

    reviews = reviews_from_json_file(args.reviews)
    pipeline = build_pipeline(load_pipeline_config(args.config))
    output = pipeline.run(PipelineInput(game=args.game, reviews=reviews))
    print(format_report(output, n_reviews=len(reviews)))
    if args.out is not None:
        written = output.to_json_file(args.out)
        print(f"\nWrote {written}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the ``srt`` argument parser."""
    parser = argparse.ArgumentParser(prog="srt", description="Steam Review Translator")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the full pipeline on a JSON list of reviews")
    demo.add_argument("--reviews", required=True, type=Path, help="JSON file: list of reviews")
    demo.add_argument("--game", required=True, help="game title")
    demo.add_argument(
        "--config", type=Path, default=None, help="pipeline config (default configs/pipeline.yaml)"
    )
    demo.add_argument("--out", type=Path, default=None, help="write PipelineOutput JSON here")
    demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``srt`` console script. Returns a process exit code."""
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (OSError, ValueError, TypeError) as exc:  # pydantic.ValidationError is a ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code
