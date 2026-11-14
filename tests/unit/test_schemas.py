from pathlib import Path

import pytest
from pydantic import ValidationError

from steam_review_translator.core.schemas import (
    Aspect,
    AspectEntry,
    AspectLabel,
    MetaReviewInput,
    PipelineInput,
    PipelineOutput,
    Polarity,
    Review,
    SentimentDistribution,
)


def _dist(**kw: object) -> SentimentDistribution:
    return SentimentDistribution.model_validate(kw)


def test_enum_values_are_the_agreed_contract() -> None:
    assert [a.value for a in Aspect] == [
        "gameplay",
        "story",
        "graphics",
        "audio",
        "performance",
        "price",
        "multiplayer",
        "difficulty",
        "content",
        "overall",
        "other",
    ]
    assert [p.value for p in Polarity] == ["positive", "negative", "neutral"]


def test_review_valid_and_optional_recommended() -> None:
    r = Review(review_id="1", text="Great game")
    assert r.recommended is None


@pytest.mark.parametrize(
    "payload",
    [
        {"review_id": "1"},  # missing text
        {"review_id": "1", "text": "x", "recommended": "maybe"},
        {"review_id": "1", "text": "x", "stars": 5},  # extra field
    ],
)
def test_review_invalid(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Review.model_validate(payload)


def test_extra_fields_forbidden_on_nested_models() -> None:
    with pytest.raises(ValidationError):
        AspectLabel.model_validate({"aspect": "story", "polarity": "positive", "score": 0.9})
    with pytest.raises(ValidationError):
        PipelineOutput.model_validate(
            {
                "game": "g",
                "sentiment": {"positive": 1, "negative": 0, "neutral": 0},
                "aspects": [],
                "meta_review": "x",
                "debug": True,
            }
        )


def test_unknown_aspect_rejected() -> None:
    with pytest.raises(ValidationError):
        AspectLabel.model_validate({"aspect": "vibes", "polarity": "positive"})


def test_aspect_entry_freq_non_negative() -> None:
    with pytest.raises(ValidationError):
        AspectEntry(aspect=Aspect.STORY, freq=-1, polarity=Polarity.NEUTRAL)


def test_pipeline_input_requires_reviews() -> None:
    with pytest.raises(ValidationError):
        PipelineInput(game="g", reviews=[])


def test_distribution_within_tolerance() -> None:
    d = _dist(positive=0.3333, negative=0.3333, neutral=0.3334)
    assert d.dominant is Polarity.NEUTRAL


def test_distribution_outside_tolerance() -> None:
    with pytest.raises(ValidationError, match="~1.0"):
        _dist(positive=0.5, negative=0.3, neutral=0.1)


def test_distribution_out_of_range() -> None:
    with pytest.raises(ValidationError):
        _dist(positive=1.2, negative=-0.2, neutral=0.0)


def test_distribution_dominant_computed() -> None:
    assert _dist(positive=0.6, negative=0.3, neutral=0.1).dominant is Polarity.POSITIVE
    assert _dist(positive=0.4, negative=0.4, neutral=0.2).dominant is Polarity.NEUTRAL


def test_distribution_dominant_validated() -> None:
    d = _dist(positive=0.6, negative=0.3, neutral=0.1, dominant="positive")
    assert d.dominant is Polarity.POSITIVE
    with pytest.raises(ValidationError, match="dominant"):
        _dist(positive=0.6, negative=0.3, neutral=0.1, dominant="negative")


def test_distribution_from_counts() -> None:
    d = SentimentDistribution.from_counts(2, 1, 1)
    assert (d.positive, d.negative, d.neutral, d.dominant) == (0.5, 0.25, 0.25, Polarity.POSITIVE)
    empty = SentimentDistribution.from_counts(0, 0, 0)
    assert empty.neutral == 1.0 and empty.dominant is Polarity.NEUTRAL


def test_json_round_trip(tmp_path: Path) -> None:
    sentiment = SentimentDistribution.from_counts(3, 1, 0)
    aspects = [AspectEntry(aspect=Aspect.GAMEPLAY, freq=3, polarity=Polarity.POSITIVE)]
    reviews = [Review(review_id="1", text="Fun combat.", recommended=True)]

    out = PipelineOutput(game="G", sentiment=sentiment, aspects=aspects, meta_review="Good.")
    path = out.to_json_file(tmp_path / "nested" / "out.json")
    assert PipelineOutput.from_json_file(path) == out

    meta = MetaReviewInput(game="G", reviews=reviews, sentiment=sentiment, aspects=aspects)
    assert MetaReviewInput.from_json_file(meta.to_json_file(tmp_path / "meta.json")) == meta


def test_meta_review_input_json_shape() -> None:
    meta = MetaReviewInput(
        game="G",
        reviews=[Review(review_id="1", text="t")],
        sentiment=SentimentDistribution.from_counts(1, 0, 0),
        aspects=[],
    )
    assert set(meta.model_dump()) == {"game", "reviews", "sentiment", "aspects"}
