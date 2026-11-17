"""Label conventions for Task 1: score -> polarity, and 3-class -> binary collapse.

Kept dependency-free so data prep, evaluation, and tests share one definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from steam_review_translator.core.schemas import Polarity

#: Fixed class order used for confusion matrices and probability vectors.
CLASSES: tuple[Polarity, ...] = (Polarity.POSITIVE, Polarity.NEGATIVE, Polarity.NEUTRAL)
#: Class order for the binary (Steam) view.
BINARY_CLASSES: tuple[Polarity, ...] = (Polarity.POSITIVE, Polarity.NEGATIVE)

#: Metacritic's own user-score colour bands: 0-4 red, 5-7 yellow, 8-10 green.
DEFAULT_NEGATIVE_MAX = 4.0
DEFAULT_POSITIVE_MIN = 8.0


def score_to_polarity(
    score: float,
    *,
    negative_max: float = DEFAULT_NEGATIVE_MAX,
    positive_min: float = DEFAULT_POSITIVE_MIN,
) -> Polarity:
    """Map a 0-10 user score to a polarity (``<= negative_max`` neg, ``>= positive_min`` pos)."""
    if not 0.0 <= score <= 10.0:
        raise ValueError(f"score must be in [0, 10], got {score}")
    if negative_max >= positive_min:
        raise ValueError("negative_max must be < positive_min")
    if score <= negative_max:
        return Polarity.NEGATIVE
    if score >= positive_min:
        return Polarity.POSITIVE
    return Polarity.NEUTRAL


def collapse_to_binary(probs: Mapping[Polarity, float]) -> Polarity:
    """Collapse a 3-class score vector to pos/neg by comparing pos vs neg (neutral ignored).

    Ties go to positive, matching Steam's majority class; ties are rare with real scores.
    """
    pos = probs.get(Polarity.POSITIVE, 0.0)
    neg = probs.get(Polarity.NEGATIVE, 0.0)
    return Polarity.POSITIVE if pos >= neg else Polarity.NEGATIVE


def recommended_to_polarity(recommended: bool) -> Polarity:
    """Steam's thumbs up/down as a polarity."""
    return Polarity.POSITIVE if recommended else Polarity.NEGATIVE


def shares(labels: Sequence[Polarity], classes: Sequence[Polarity] = CLASSES) -> list[float]:
    """Fraction of ``labels`` per class, in ``classes`` order. Empty input gives all zeros."""
    n = len(labels)
    if n == 0:
        return [0.0] * len(classes)
    return [sum(1 for lab in labels if lab == c) / n for c in classes]
