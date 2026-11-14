"""The shared data contract between the three tasks (pydantic v2).

Field names here are the agreed JSON contract. Changing them is a breaking change for
every task owner; see ``docs/contributing.md`` before editing this file.
"""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, TypeAdapter, model_validator

#: Allowed absolute deviation of ``positive + negative + neutral`` from 1.0.
SUM_TOLERANCE = 1e-3


class Aspect(StrEnum):
    """Closed set of review aspects. ``other`` is the fallback for unmapped mentions."""

    GAMEPLAY = "gameplay"
    STORY = "story"
    GRAPHICS = "graphics"
    AUDIO = "audio"
    PERFORMANCE = "performance"
    PRICE = "price"
    MULTIPLAYER = "multiplayer"
    DIFFICULTY = "difficulty"
    CONTENT = "content"
    OVERALL = "overall"
    OTHER = "other"


class Polarity(StrEnum):
    """Sentiment polarity label."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ContractModel(BaseModel):
    """Base for all contract models: rejects unknown fields and adds JSON file helpers."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_json_file(cls, path: str | Path) -> Self:
        """Load and validate an instance from a UTF-8 JSON file."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_json_file(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write this instance as JSON (creating parent dirs) and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=indent) + "\n", encoding="utf-8")
        return out


class Review(ContractModel):
    """A single Steam review. ``recommended`` is Steam's thumbs up/down, if known."""

    review_id: str
    text: str
    recommended: bool | None = None


class AspectLabel(ContractModel):
    """One aspect mention in one sentence, with the polarity expressed towards it."""

    aspect: Aspect
    polarity: Polarity


def _dominant(scores: dict[Polarity, float]) -> set[Polarity]:
    """Return every polarity that is acceptable as dominant (ties also allow neutral)."""
    best = max(scores.values())
    maxima = {p for p, v in scores.items() if math.isclose(v, best, abs_tol=1e-9)}
    if len(maxima) > 1:
        maxima.add(Polarity.NEUTRAL)
    return maxima


def pick_dominant(scores: dict[Polarity, float]) -> Polarity:
    """Pick the single dominant polarity; ties resolve to neutral."""
    candidates = _dominant(scores)
    if len(candidates) == 1:
        return next(iter(candidates))
    return Polarity.NEUTRAL


class SentimentDistribution(ContractModel):
    """Task 1 output: share of reviews per polarity (sums to ~1) and the dominant label.

    ``dominant`` may be omitted on input, in which case it is computed (ties -> neutral).
    If given, it must be one of the maximal polarities.
    """

    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    dominant: Polarity

    @model_validator(mode="before")
    @classmethod
    def _fill_dominant(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("dominant") is None:
            try:
                scores = {p: float(data[p.value]) for p in Polarity}
            except (KeyError, TypeError, ValueError):
                return data  # let normal field validation report the problem
            data = {**data, "dominant": pick_dominant(scores)}
        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        total = self.positive + self.negative + self.neutral
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise ValueError(f"positive + negative + neutral must be ~1.0, got {total:.6f}")
        if self.dominant not in _dominant(self.as_dict()):
            raise ValueError(f"dominant={self.dominant.value!r} is not a maximal polarity")
        return self

    def as_dict(self) -> dict[Polarity, float]:
        """Return the three shares keyed by :class:`Polarity`."""
        return {
            Polarity.POSITIVE: self.positive,
            Polarity.NEGATIVE: self.negative,
            Polarity.NEUTRAL: self.neutral,
        }

    @classmethod
    def from_counts(cls, positive: int, negative: int, neutral: int) -> SentimentDistribution:
        """Build a distribution from raw counts. All-zero counts give 100% neutral."""
        total = positive + negative + neutral
        if total == 0:
            return cls(positive=0.0, negative=0.0, neutral=1.0, dominant=Polarity.NEUTRAL)
        scores = {
            Polarity.POSITIVE: positive / total,
            Polarity.NEGATIVE: negative / total,
            Polarity.NEUTRAL: neutral / total,
        }
        return cls(
            positive=scores[Polarity.POSITIVE],
            negative=scores[Polarity.NEGATIVE],
            neutral=scores[Polarity.NEUTRAL],
            dominant=pick_dominant(scores),
        )


class AspectEntry(ContractModel):
    """One row of the Task 2 aspect table."""

    aspect: Aspect
    freq: NonNegativeInt
    polarity: Polarity


class PipelineInput(ContractModel):
    """A batch of reviews for one game."""

    game: str
    reviews: list[Review] = Field(min_length=1)


class MetaReviewInput(ContractModel):
    """The exact JSON payload handed to Task 3."""

    game: str
    reviews: list[Review] = Field(min_length=1)
    sentiment: SentimentDistribution
    aspects: list[AspectEntry]


class PipelineOutput(ContractModel):
    """Final pipeline result for one game."""

    game: str
    sentiment: SentimentDistribution
    aspects: list[AspectEntry]
    meta_review: str


_REVIEW_LIST = TypeAdapter(list[Review])


def reviews_from_json_file(path: str | Path) -> list[Review]:
    """Load a JSON file containing a list of :class:`Review` objects."""
    return _REVIEW_LIST.validate_json(Path(path).read_text(encoding="utf-8"))
