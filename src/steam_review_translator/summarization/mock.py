"""Deterministic stand-in for Task 3. No ML: replace with a real ``MetaReviewGenerator``."""

from __future__ import annotations

from steam_review_translator.core.schemas import AspectEntry, MetaReviewInput, Polarity


class MockMetaReviewGenerator:
    """Fill a fixed template from the sentiment distribution and aspect table."""

    def __init__(self, top_k: int = 3) -> None:
        """``top_k``: how many of the most frequent aspects to mention."""
        self.top_k = top_k

    def generate(self, inp: MetaReviewInput) -> str:
        """Return a one-paragraph meta-review."""
        s = inp.sentiment
        parts = [
            f"Across {len(inp.reviews)} reviews, players' reception of {inp.game} is mostly "
            f"{s.dominant.value} ({s.positive:.0%} positive, {s.negative:.0%} negative, "
            f"{s.neutral:.0%} neutral)."
        ]
        top = [e for e in inp.aspects if e.freq > 0][: self.top_k]
        if top:
            listed = _join([f"{e.aspect.value} ({e.freq} mentions)" for e in top])
            parts.append(f"The most discussed aspects are {listed}.")
        praised = _names(inp.aspects, Polarity.POSITIVE)
        criticised = _names(inp.aspects, Polarity.NEGATIVE)
        if praised:
            parts.append(f"Reviewers mostly praise the {praised}.")
        if criticised:
            parts.append(f"Common complaints concern the {criticised}.")
        if not top:
            parts.append("No specific aspects stood out in these reviews.")
        return " ".join(parts)


def _names(entries: list[AspectEntry], polarity: Polarity) -> str:
    return _join([e.aspect.value for e in entries if e.polarity is polarity and e.freq > 0])


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"
