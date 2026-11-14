"""Turn per-sentence aspect labels into the Task 2 aspect table."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from steam_review_translator.core.schemas import Aspect, AspectEntry, AspectLabel, Polarity


def aggregate(labels: Sequence[Sequence[AspectLabel]]) -> list[AspectEntry]:
    """Build the aspect table from ``AspectTagger.tag`` output (one list per sentence).

    - ``freq``: number of sentences mentioning the aspect. If one sentence carries the
      same aspect more than once, only its first label counts.
    - ``polarity``: majority polarity over those sentences; a tie for the top -> neutral.
    - Order: ``freq`` descending, then aspect name ascending (deterministic).
    """
    votes: dict[Aspect, Counter[Polarity]] = {}
    for sentence_labels in labels:
        seen: set[Aspect] = set()
        for label in sentence_labels:
            if label.aspect in seen:
                continue
            seen.add(label.aspect)
            votes.setdefault(label.aspect, Counter())[label.polarity] += 1

    table = [
        AspectEntry(aspect=aspect, freq=sum(c.values()), polarity=_majority(c))
        for aspect, c in votes.items()
    ]
    table.sort(key=lambda e: (-e.freq, e.aspect.value))
    return table


def _majority(counts: Counter[Polarity]) -> Polarity:
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return Polarity.NEUTRAL
    return ranked[0][0]
