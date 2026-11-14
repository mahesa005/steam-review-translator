"""Topic -> aspect mapping (e.g. BERTopic topics onto the fixed :class:`Aspect` set).

Stub status: the decision rule below is a placeholder. TODO(Mahesa): define how per-topic
aspect scores are computed (lexicon overlap, embedding similarity, ...) and tune
``floor`` / ``ratio`` on the dev set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from steam_review_translator.core.config import load_yaml
from steam_review_translator.core.schemas import Aspect

DEFAULT_MAPPING = "configs/task2/mapping.yaml"


@dataclass(frozen=True)
class MappingParams:
    """Thresholds for assigning a topic to an aspect.

    ``floor``: minimum score the best aspect must reach.
    ``ratio``: minimum share of the topic's total score mass the best aspect must hold.
    """

    floor: float = 0.3
    ratio: float = 0.5

    def __post_init__(self) -> None:
        """Validate that both thresholds lie in [0, 1]."""
        for name in ("floor", "ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MappingParams:
        """Build from a mapping with optional ``floor`` and ``ratio`` keys."""
        return cls(floor=float(data.get("floor", 0.3)), ratio=float(data.get("ratio", 0.5)))


def load_mapping_params(path: str | Path | None = None) -> MappingParams:
    """Load thresholds, defaulting to ``configs/task2/mapping.yaml``."""
    return MappingParams.from_dict(load_yaml(path or DEFAULT_MAPPING))


def map_topic(scores: Mapping[Aspect, float], params: MappingParams) -> Aspect:
    """Map one topic's aspect scores to a single aspect, or ``Aspect.OTHER``."""
    if not scores:
        return Aspect.OTHER
    best_aspect, best = max(scores.items(), key=lambda kv: (kv[1], kv[0].value))
    total = sum(max(v, 0.0) for v in scores.values())
    if best < params.floor or total <= 0 or best / total < params.ratio:
        return Aspect.OTHER
    return best_aspect


def map_topics(
    topic_scores: Mapping[int, Mapping[Aspect, float]], params: MappingParams
) -> dict[int, Aspect]:
    """Apply :func:`map_topic` to every topic id."""
    return {topic: map_topic(scores, params) for topic, scores in topic_scores.items()}
