"""Shared plumbing: a review classifier + counting aggregation = a ``SentimentAnalyzer``.

Every Task 1 system (S0-S4) only has to implement :meth:`ReviewClassifier.scores`;
aggregation is identical for all of them, as the experiment design requires.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from steam_review_translator.core.schemas import Polarity, Review, SentimentDistribution
from steam_review_translator.sentiment.labels import CLASSES, collapse_to_binary

Scores = dict[Polarity, float]


class ReviewClassifier(ABC):
    """Labels individual reviews. Subclasses implement :meth:`scores`.

    Implementations must never look at ``Review.recommended``: on Steam it is the gold
    label, so reading it would leak the answer.
    """

    #: Short system id used in result files (e.g. ``"S0-vader"``).
    name: str = "classifier"

    @abstractmethod
    def scores(self, texts: Sequence[str]) -> list[Scores]:
        """Return one score per class per text (higher = more likely). Need not sum to 1."""

    def label_from_scores(self, scores: Scores) -> Polarity:
        """3-class decision rule. Default: argmax (ties resolved in ``CLASSES`` order)."""
        return max(CLASSES, key=lambda c: scores.get(c, 0.0))

    def predict(self, texts: Sequence[str]) -> list[Polarity]:
        """3-class labels via :meth:`label_from_scores`."""
        return [self.label_from_scores(s) for s in self.scores(texts)]

    def predict_binary(self, texts: Sequence[str]) -> list[Polarity]:
        """pos/neg labels: compare the positive and negative scores, ignore neutral."""
        return [collapse_to_binary(s) for s in self.scores(texts)]

    def analyze(self, reviews: list[Review]) -> SentimentDistribution:
        """``SentimentAnalyzer`` protocol: classify every review, then count."""
        if not reviews:
            return SentimentDistribution.from_counts(0, 0, 0)
        labels = self.predict([r.text for r in reviews])
        return SentimentDistribution.from_counts(
            labels.count(Polarity.POSITIVE),
            labels.count(Polarity.NEGATIVE),
            labels.count(Polarity.NEUTRAL),
        )
