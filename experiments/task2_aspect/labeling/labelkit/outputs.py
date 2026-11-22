"""Output contracts of the three stages: JSON schemas sent to the API + strict validation.

The JSON schemas constrain what the model can return (structured outputs); the pydantic models
re-check every parsed response and ``check_ids`` makes sure no sentence is missing, duplicated
or invented. Aspect and polarity values come from the library contract
(``steam_review_translator.core.schemas``), so labels stay compatible with ``AspectLabel``.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from steam_review_translator.core.schemas import Aspect, Polarity

ASPECT_VALUES = [a.value for a in Aspect]
POLARITY_VALUES = [p.value for p in Polarity]


class Verdict(StrEnum):
    """Reviewer verdict for one sentence."""

    OK = "ok"
    FLAG = "flag"


class Issue(StrEnum):
    """Why a reviewer flagged a sentence (``none`` when the verdict is ``ok``)."""

    NONE = "none"
    WRONG_ASPECT = "wrong_aspect"
    MISSING_ASPECT = "missing_aspect"
    EXTRA_ASPECT = "extra_aspect"
    WRONG_POLARITY = "wrong_polarity"
    GUIDELINE = "guideline"


class BatchVerdict(StrEnum):
    """Reviewer verdict for the whole batch."""

    PASS = "pass"
    NEEDS_REVISION = "needs_revision"


class Decision(StrEnum):
    """Stage-3 outcome for one flagged sentence."""

    KEPT = "kept"
    CHANGED = "changed"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Label(_Strict):
    """One (aspect, polarity) pair."""

    aspect: Aspect
    polarity: Polarity


class SentenceLabels(_Strict):
    """Stage 1: the labels of one sentence."""

    sentence_id: str
    labels: list[Label]


class LabelOutput(_Strict):
    """Stage 1 response."""

    sentences: list[SentenceLabels]


class ReviewItem(_Strict):
    """Stage 2: one reviewer's judgment of one sentence."""

    sentence_id: str
    verdict: Verdict
    issue: Issue
    reasoning: str
    suggested_labels: list[Label]


class ReviewOutput(_Strict):
    """Stage 2 response (one reviewer)."""

    items: list[ReviewItem]
    batch_verdict: BatchVerdict
    notes: str


class FinalItem(_Strict):
    """Stage 3: the final decision for one flagged sentence."""

    sentence_id: str
    final_labels: list[Label]
    decision: Decision
    rationale: str


class FinalizeOutput(_Strict):
    """Stage 3 response."""

    decisions: list[FinalItem]


# ---------------------------------------------------------------------------------------
# JSON schemas for output_config.format (every object: all fields required, no extras)
# ---------------------------------------------------------------------------------------
def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def _enum(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


_LABEL = _obj({"aspect": _enum(ASPECT_VALUES), "polarity": _enum(POLARITY_VALUES)})
_LABELS = {"type": "array", "items": _LABEL}

LABEL_SCHEMA = _obj(
    {
        "sentences": {
            "type": "array",
            "items": _obj({"sentence_id": {"type": "string"}, "labels": _LABELS}),
        }
    }
)
REVIEW_SCHEMA = _obj(
    {
        "items": {
            "type": "array",
            "items": _obj(
                {
                    "sentence_id": {"type": "string"},
                    "verdict": _enum([v.value for v in Verdict]),
                    "issue": _enum([i.value for i in Issue]),
                    "reasoning": {"type": "string"},
                    "suggested_labels": _LABELS,
                }
            ),
        },
        "batch_verdict": _enum([v.value for v in BatchVerdict]),
        "notes": {"type": "string"},
    }
)
FINALIZE_SCHEMA = _obj(
    {
        "decisions": {
            "type": "array",
            "items": _obj(
                {
                    "sentence_id": {"type": "string"},
                    "final_labels": _LABELS,
                    "decision": _enum([d.value for d in Decision]),
                    "rationale": {"type": "string"},
                }
            ),
        }
    }
)


def check_ids(got: Sequence[str], expected: Sequence[str]) -> None:
    """Raise ``ValueError`` unless ``got`` covers ``expected`` exactly once each."""
    dupes = sorted({i for i in got if list(got).count(i) > 1})
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    if dupes or missing or extra:
        raise ValueError(f"sentence ids: missing={missing} extra={extra} duplicated={dupes}")


def dedupe_labels(labels: Sequence[Label]) -> tuple[list[Label], bool]:
    """Enforce one label per aspect (guideline rule 2): conflicting duplicates -> neutral.

    Returns the cleaned labels and whether anything had to be changed.
    """
    by_aspect: dict[Aspect, Polarity] = {}
    changed = False
    for label in labels:
        if label.aspect in by_aspect:
            changed = True
            if by_aspect[label.aspect] != label.polarity:
                by_aspect[label.aspect] = Polarity.NEUTRAL
        else:
            by_aspect[label.aspect] = label.polarity
    return [Label(aspect=a, polarity=p) for a, p in by_aspect.items()], changed
