"""Task interfaces. Real implementations only need to match these method signatures.

The pipeline depends on these Protocols, never on concrete classes, so each owner can
swap in a real model without touching anyone else's code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from steam_review_translator.core.schemas import (
    AspectLabel,
    MetaReviewInput,
    Review,
    SentimentDistribution,
)


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Task 1 (owner: Farrel)."""

    def analyze(self, reviews: list[Review]) -> SentimentDistribution:
        """Classify every review and return the share per polarity over the whole batch.

        Must be deterministic for a fixed model/config and must not mutate ``reviews``.
        An empty list should return 100% neutral rather than raising.
        """
        ...


@runtime_checkable
class AspectTagger(Protocol):
    """Task 2 (owner: Mahesa)."""

    def tag(self, sentences: list[str]) -> list[list[AspectLabel]]:
        """Return one list of aspect labels per input sentence, in the same order.

        ``len(result) == len(sentences)`` always holds; a sentence with no aspect gets
        ``[]``. The table is built afterwards by ``aspect.aggregate.aggregate``.
        """
        ...


@runtime_checkable
class MetaReviewGenerator(Protocol):
    """Task 3 (owner: Lukas)."""

    def generate(self, inp: MetaReviewInput) -> str:
        """Write a single-paragraph meta-review for ``inp.game``.

        Should be grounded in ``inp.aspects`` (and may quote ``inp.reviews``). Returns a
        non-empty string.
        """
        ...
