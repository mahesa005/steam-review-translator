"""Tests for the Task 2 labeling helpers (experiments/task2_aspect/labeling/labelkit).

No API calls: these cover batching, output validation and request building only.
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

LABELING_DIR = Path(__file__).resolve().parents[2] / "experiments" / "task2_aspect" / "labeling"
sys.path.insert(0, str(LABELING_DIR))

from labelkit.batches import SampleSentence, make_batches  # noqa: E402
from labelkit.outputs import (  # noqa: E402
    LABEL_SCHEMA,
    Label,
    LabelOutput,
    ReviewOutput,
    check_ids,
    dedupe_labels,
)
from labelkit.requests import Prompts, build_params, custom_id, render_batch  # noqa: E402


def rows_for(game_id: int, review_id: int, n: int, split: str = "train") -> list[SampleSentence]:
    return [
        SampleSentence(
            game_id, f"Game {game_id}", split, review_id, f"{review_id}-{k}", f"s{k}.", True
        )
        for k in range(n)
    ]


def texts(*review_ids: int) -> dict[int, str]:
    return {r: f"Full text of review {r}." for r in review_ids}


# --------------------------------------------------------------------------- batching
def test_batches_keep_reviews_whole_and_respect_size() -> None:
    rows = rows_for(1, 10, 30) + rows_for(1, 11, 15) + rows_for(1, 12, 10)
    batches = make_batches(rows, texts(10, 11, 12), max_sentences=50)
    assert [b.batch_id for b in batches] == ["1-000", "1-001"]
    assert [[r.review_id for r in b.reviews] for b in batches] == [[10, 11], [12]]
    assert [b.n_sentences for b in batches] == [45, 10]


def test_oversized_review_gets_its_own_batch() -> None:
    rows = rows_for(1, 10, 5) + rows_for(1, 11, 80) + rows_for(1, 12, 5)
    batches = make_batches(rows, texts(10, 11, 12), max_sentences=50)
    assert [b.n_sentences for b in batches] == [5, 80, 5]


def test_batches_are_per_game_and_deterministic() -> None:
    rows = rows_for(2, 20, 3) + rows_for(1, 10, 3)
    a = make_batches(rows, texts(10, 20))
    b = make_batches(list(reversed(rows)), texts(10, 20))
    assert [x.batch_id for x in a] == ["1-000", "2-000"]
    assert [x.sentence_ids for x in a] == [x.sentence_ids for x in b]  # order-independent


def test_sentence_order_within_review_is_by_index() -> None:
    rows = rows_for(1, 10, 12)
    batches = make_batches(list(reversed(rows)), texts(10))
    assert batches[0].sentence_ids == [f"10-{k}" for k in range(12)]  # 10-2 before 10-10


def test_missing_review_text_is_an_error() -> None:
    with pytest.raises(KeyError):
        make_batches(rows_for(1, 10, 2), {})


# --------------------------------------------------------------------------- outputs
def test_label_output_parses_and_rejects_bad_values() -> None:
    ok = {
        "sentences": [
            {"sentence_id": "1-0", "labels": [{"aspect": "story", "polarity": "positive"}]}
        ]
    }
    assert LabelOutput.model_validate(ok).sentences[0].labels[0].aspect.value == "story"
    bad = {
        "sentences": [
            {"sentence_id": "1-0", "labels": [{"aspect": "vibes", "polarity": "positive"}]}
        ]
    }
    with pytest.raises(ValidationError):
        LabelOutput.model_validate(bad)


def test_review_output_requires_all_fields() -> None:
    item = {
        "sentence_id": "1-0",
        "verdict": "ok",
        "issue": "none",
        "reasoning": "",
        "suggested_labels": [],
    }
    ReviewOutput.model_validate({"items": [item], "batch_verdict": "pass", "notes": ""})
    with pytest.raises(ValidationError):
        ReviewOutput.model_validate({"items": [item], "batch_verdict": "pass"})


def test_check_ids() -> None:
    check_ids(["a", "b"], ["b", "a"])
    with pytest.raises(ValueError, match="missing"):
        check_ids(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="duplicated"):
        check_ids(["a", "a", "b"], ["a", "b"])


def test_dedupe_labels_conflict_becomes_neutral() -> None:
    labels = [
        Label(aspect="story", polarity="positive"),
        Label(aspect="story", polarity="negative"),
        Label(aspect="audio", polarity="positive"),
    ]
    clean, changed = dedupe_labels(labels)
    assert changed
    assert [(x.aspect.value, x.polarity.value) for x in clean] == [
        ("story", "neutral"),
        ("audio", "positive"),
    ]


def test_schema_is_strict() -> None:
    def walk(node: dict) -> None:
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])

    walk(LABEL_SCHEMA)
    json.dumps(LABEL_SCHEMA)  # serializable


# --------------------------------------------------------------------------- requests
@pytest.fixture(scope="module")
def prompts() -> Prompts:
    return Prompts.load(LABELING_DIR / "prompts")


def test_render_batch_lists_every_sentence_with_review(prompts: Prompts) -> None:
    batch = make_batches(rows_for(1, 10, 2) + rows_for(1, 11, 1), texts(10, 11))[0]
    text = render_batch(batch)
    assert "<game>Game 1</game>" in text
    assert text.count("<review id=") == 2
    for sid in batch.sentence_ids:
        assert f"[{sid}]" in text


def test_render_batch_with_labels_and_only_ids() -> None:
    batch = make_batches(rows_for(1, 10, 2) + rows_for(1, 11, 1), texts(10, 11))[0]
    labels = {sid: [Label(aspect="gameplay", polarity="negative")] for sid in batch.sentence_ids}
    text = render_batch(batch, labels=labels, only_ids={"11-0"})
    assert "[11-0]" in text and "[10-0]" not in text
    assert '<review id="10"' not in text  # reviews with no listed sentence are left out
    assert "labels: gameplay/negative" in text


def test_render_escapes_markup() -> None:
    rows = [SampleSentence(1, "A & B", "train", 10, "10-0", "x < y", True)]
    text = render_batch(make_batches(rows, {10: "<b>bold</b>"})[0])
    assert "A &amp; B" in text and "x &lt; y" in text and "&lt;b&gt;" in text


def test_build_params_shape(prompts: Prompts) -> None:
    cfg = {"model": "claude-opus-5-5", "effort": "medium", "max_tokens": 32000}
    p = build_params("label", prompts, "hello", cfg)
    assert p["model"] == "claude-opus-5-5"
    assert p["output_config"]["effort"] == "medium"
    assert p["output_config"]["format"]["schema"] is LABEL_SCHEMA
    assert p["system"][0]["text"] == prompts.guideline
    assert p["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "budget_tokens" not in p["thinking"]


def test_custom_id_format() -> None:
    import re

    for cid in (custom_id("label", "289690-003"), custom_id("review", "289690-003", reviewer=2)):
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid)
    assert custom_id("review", "1-000", reviewer=2) == "review2-1-000"


def test_prompt_hash_is_stable(prompts: Prompts) -> None:
    assert prompts.sha256 == Prompts.load(LABELING_DIR / "prompts").sha256
