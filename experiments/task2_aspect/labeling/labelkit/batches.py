"""Group the labeling sample into batches of ~50 sentences, never splitting a review."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

MAX_SENTENCES = 50


@dataclass(frozen=True)
class SampleSentence:
    """One row of data/processed/candidate_sample.jsonl."""

    game_id: int
    game_name: str
    split: str
    review_id: int
    sentence_id: str
    sentence: str
    recommended: bool

    @property
    def index(self) -> int:
        """0-based position of the sentence within its review (from ``<review_id>-<k>``)."""
        return int(self.sentence_id.rsplit("-", 1)[1])


@dataclass
class Review:
    """A review with its sentences (in order) and its full text for context."""

    review_id: int
    recommended: bool
    text: str
    sentences: list[SampleSentence] = field(default_factory=list)


@dataclass
class Batch:
    """One unit of work for every stage: whole reviews of one game, ~50 sentences."""

    batch_id: str
    game_id: int
    game_name: str
    split: str
    reviews: list[Review]

    @property
    def sentence_ids(self) -> list[str]:
        """All sentence ids in batch order."""
        return [s.sentence_id for r in self.reviews for s in r.sentences]

    @property
    def n_sentences(self) -> int:
        """Number of sentences to label in this batch."""
        return sum(len(r.sentences) for r in self.reviews)


def make_batches(
    rows: Iterable[SampleSentence],
    review_texts: dict[int, str],
    max_sentences: int = MAX_SENTENCES,
) -> list[Batch]:
    """Build deterministic batches.

    Games are processed in ``game_id`` order and reviews in ``review_id`` order. Whole reviews
    are added to the current batch while it stays at or under ``max_sentences``; a review
    longer than ``max_sentences`` gets a batch of its own. Batch ids are
    ``<game_id>-<k>`` with ``k`` counting from 000 within each game.
    """
    by_game: dict[int, dict[int, Review]] = {}
    meta: dict[int, tuple[str, str]] = {}
    for row in rows:
        meta.setdefault(row.game_id, (row.game_name, row.split))
        reviews = by_game.setdefault(row.game_id, {})
        if row.review_id not in reviews:
            if row.review_id not in review_texts:
                raise KeyError(f"no review text for review_id {row.review_id}")
            reviews[row.review_id] = Review(
                row.review_id, row.recommended, review_texts[row.review_id]
            )
        reviews[row.review_id].sentences.append(row)

    batches: list[Batch] = []
    for game_id in sorted(by_game):
        name, split = meta[game_id]
        groups: list[list[Review]] = [[]]
        count = 0
        for review_id in sorted(by_game[game_id]):
            review = by_game[game_id][review_id]
            review.sentences.sort(key=lambda s: s.index)
            if groups[-1] and count + len(review.sentences) > max_sentences:
                groups.append([])
                count = 0
            groups[-1].append(review)
            count += len(review.sentences)
        for k, group in enumerate(g for g in groups if g):
            batches.append(Batch(f"{game_id}-{k:03d}", game_id, name, split, group))
    return batches
