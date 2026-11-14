"""Deterministic stand-in for Task 1. No ML: replace with a real ``SentimentAnalyzer``."""

from __future__ import annotations

import re

from steam_review_translator.core.schemas import Polarity, Review, SentimentDistribution

_WORD_RE = re.compile(r"[a-z']+")
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
        "beautiful",
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


class MockSentimentAnalyzer:
    """Label each review, then return the batch distribution.

    Rule: use Steam's ``recommended`` flag when present; otherwise count positive vs.
    negative keywords (no words or a tie -> neutral).
    """

    def analyze(self, reviews: list[Review]) -> SentimentDistribution:
        """Return the share of positive / negative / neutral reviews."""
        counts = dict.fromkeys(Polarity, 0)
        for review in reviews:
            counts[self._label(review)] += 1
        return SentimentDistribution.from_counts(
            counts[Polarity.POSITIVE], counts[Polarity.NEGATIVE], counts[Polarity.NEUTRAL]
        )

    @staticmethod
    def _label(review: Review) -> Polarity:
        if review.recommended is not None:
            return Polarity.POSITIVE if review.recommended else Polarity.NEGATIVE
        words = _WORD_RE.findall(review.text.lower())
        score = sum(w in _POSITIVE for w in words) - sum(w in _NEGATIVE for w in words)
        if score > 0:
            return Polarity.POSITIVE
        if score < 0:
            return Polarity.NEGATIVE
        return Polarity.NEUTRAL
