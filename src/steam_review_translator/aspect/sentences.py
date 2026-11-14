"""Sentence splitting used by the pipeline before aspect tagging."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` on ``.``/``!``/``?`` followed by whitespace, and on newlines.

    TODO(Mahesa): replace with spaCy's sentencizer (``.[task2]``); this regex breaks on
    abbreviations like "e.g." and ignores emoji/ellipsis-only reviews.
    """
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
