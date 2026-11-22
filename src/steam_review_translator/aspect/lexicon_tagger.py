"""System A: lexicon aspect detection + VADER sentence polarity.

Aspects come from whole-token lexicon hits (default: the seed lexicon grown with terms
induced from the train labels, ``configs/task2/lexicon_induced.yaml``). Every aspect found
in a sentence gets the sentence's VADER polarity: compound >= ``pos`` -> positive,
<= ``neg`` -> negative, otherwise neutral. vaderSentiment is imported lazily.
"""

from __future__ import annotations

from pathlib import Path

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.schemas import AspectLabel, Polarity

DEFAULT_LEXICON = "configs/task2/lexicon_induced.yaml"


class LexiconAspectTagger:
    """Lexicon hits for aspects, VADER for polarity."""

    def __init__(
        self,
        lexicon: str | Path = DEFAULT_LEXICON,
        pos: float | str = 0.05,
        neg: float | str = -0.05,
    ) -> None:
        """Load the lexicon and the VADER analyzer."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._matcher = LexiconMatcher(load_lexicon(lexicon))
        self._vader = SentimentIntensityAnalyzer()
        self._pos, self._neg = float(pos), float(neg)

    def polarity(self, sentence: str) -> Polarity:
        """VADER compound score mapped to a polarity."""
        c = self._vader.polarity_scores(sentence)["compound"]
        if c >= self._pos:
            return Polarity.POSITIVE
        if c <= self._neg:
            return Polarity.NEGATIVE
        return Polarity.NEUTRAL

    def tag(self, sentences: list[str]) -> list[list[AspectLabel]]:
        """Return one list of labels per sentence."""
        out = []
        for s in sentences:
            aspects = self._matcher.match(s)
            pol = self.polarity(s) if aspects else Polarity.NEUTRAL
            out.append([AspectLabel(aspect=a, polarity=pol) for a in aspects])
        return out
