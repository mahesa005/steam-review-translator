"""S0: VADER lexicon baseline (needs the ``task1`` extra: ``vaderSentiment``)."""

from __future__ import annotations

from collections.abc import Sequence

from steam_review_translator.core.schemas import Polarity
from steam_review_translator.sentiment.base import ReviewClassifier, Scores


class VaderSentimentAnalyzer(ReviewClassifier):
    """VADER compound score with the standard +/-0.05 thresholds.

    ``scores`` exposes the compound score split into pos/neg parts, so the binary
    collapse (pos vs neg) is simply the sign of the compound score.
    """

    name = "S0-vader"

    def __init__(self, threshold: float = 0.05) -> None:
        """``threshold``: |compound| below this is neutral."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.threshold = threshold
        self._analyzer = SentimentIntensityAnalyzer()

    def _compound(self, text: str) -> float:
        return float(self._analyzer.polarity_scores(text)["compound"])

    def scores(self, texts: Sequence[str]) -> list[Scores]:
        """pos = max(compound, 0), neg = max(-compound, 0), neu = 1 - |compound|."""
        out: list[Scores] = []
        for text in texts:
            c = self._compound(text)
            out.append(
                {
                    Polarity.POSITIVE: max(c, 0.0),
                    Polarity.NEGATIVE: max(-c, 0.0),
                    Polarity.NEUTRAL: 1.0 - abs(c),
                }
            )
        return out

    def label_from_scores(self, scores: Scores) -> Polarity:
        """Standard VADER rule on compound (= pos - neg): >= t pos, <= -t neg, else neutral."""
        c = scores[Polarity.POSITIVE] - scores[Polarity.NEGATIVE]
        if c >= self.threshold:
            return Polarity.POSITIVE
        if c <= -self.threshold:
            return Polarity.NEGATIVE
        return Polarity.NEUTRAL
