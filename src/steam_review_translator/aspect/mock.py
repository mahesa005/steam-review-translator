"""Deterministic stand-in for Task 2. No ML: replace with a real ``AspectTagger``."""

from __future__ import annotations

from pathlib import Path

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon, tokenize
from steam_review_translator.core.schemas import AspectLabel, Polarity

_POSITIVE = frozenset(
    {
        "good",
        "great",
        "amazing",
        "awesome",
        "love",
        "loved",
        "loves",
        "lovely",
        "fun",
        "excellent",
        "fantastic",
        "gorgeous",
        "solid",
        "enjoy",
        "enjoyed",
        "best",
        "perfect",
        "masterpiece",
        "recommend",
        "incredible",
        "clever",
        "fast",
    }
)
_NEGATIVE = frozenset(
    {
        "bad",
        "terrible",
        "awful",
        "boring",
        "hate",
        "broken",
        "worst",
        "poor",
        "crash",
        "crashes",
        "lag",
        "stutter",
        "expensive",
        "disappointing",
        "annoying",
        "overpriced",
        "refund",
        "repetitive",
        "grindy",
        "unplayable",
        "mediocre",
    }
)


class MockAspectTagger:
    """Lexicon keyword hits for aspects; sentence-level keyword vote for polarity.

    Every aspect found in a sentence gets that sentence's polarity.
    """

    def __init__(self, lexicon: str | Path | None = None) -> None:
        """Load the lexicon (default ``configs/task2/lexicon.yaml``)."""
        self._matcher = LexiconMatcher(load_lexicon(lexicon))

    def tag(self, sentences: list[str]) -> list[list[AspectLabel]]:
        """Return one list of labels per sentence."""
        return [self._tag_one(s) for s in sentences]

    def _tag_one(self, sentence: str) -> list[AspectLabel]:
        aspects = self._matcher.match(sentence)
        if not aspects:
            return []
        polarity = _sentence_polarity(sentence)
        return [AspectLabel(aspect=a, polarity=polarity) for a in aspects]


def _sentence_polarity(sentence: str) -> Polarity:
    tokens = tokenize(sentence)
    score = sum(t in _POSITIVE for t in tokens) - sum(t in _NEGATIVE for t in tokens)
    if score > 0:
        return Polarity.POSITIVE
    if score < 0:
        return Polarity.NEGATIVE
    return Polarity.NEUTRAL
