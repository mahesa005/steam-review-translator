"""Thin shim: ``python demo.py --reviews reviews.json --game "Game Name"`` == ``srt demo ...``."""

import sys
from pathlib import Path

# Allow running from a fresh clone before `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from steam_review_translator.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["demo", *sys.argv[1:]]))
