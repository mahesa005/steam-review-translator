"""Unit tests for Task 3 (summarization) - owner: Lukas.

Semua test ini jalan tanpa extra berat (tidak butuh torch/transformers), supaya CI
yang hanya memasang ``.[dev]`` tetap hijau.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.schemas import (
    Aspect,
    AspectEntry,
    MetaReviewInput,
    PipelineOutput,
    Polarity,
    Review,
)
from steam_review_translator.summarization.aspects import pseudo_aspect_table
from steam_review_translator.summarization.extractive import (
    ExtractiveMetaReviewGenerator,
    lead_k,
    textrank,
)
from steam_review_translator.summarization.metrics import (
    aspect_coverage,
    average_rouge,
    paired_bootstrap,
    polarity_consistency,
    rouge_scores,
)
from steam_review_translator.summarization.preprocessing import (
    BoilerplateFilter,
    Paragraph,
    ParagraphSelector,
    preprocess_document,
)
from steam_review_translator.summarization.serialization import (
    build_inference_input,
    format_aspect_prefix,
    serialize_input,
)


@pytest.fixture(scope="module")
def matcher() -> LexiconMatcher:
    """Matcher lexicon Task 2 yang dipakai Task 3."""
    return LexiconMatcher(load_lexicon())


def _review(text: str, index: int = 0) -> Review:
    """Bantu: buat objek Review."""
    return Review(review_id=f"r{index:03d}", text=text, recommended=None)


def _meta_input(reviews: list[Review], aspects: list[AspectEntry]) -> MetaReviewInput:
    """Bantu: rakit MetaReviewInput (kontrak Task 3)."""
    return MetaReviewInput(
        game="Fake Game",
        reviews=reviews,
        sentiment={
            "positive": 0.5,
            "negative": 0.3,
            "neutral": 0.2,
            "dominant": "positive",
        },
        aspects=aspects,
    )


# --------------------------------------------------------------------------- [A] praproses
def test_boilerplate_filter_drops_known_noise() -> None:
    cleaner = BoilerplateFilter()
    assert cleaner.is_boilerplate("Please enable JavaScript to continue browsing this site")
    assert cleaner.is_boilerplate("All rights reserved. Copyright 2024.")
    assert not cleaner.is_boilerplate("The combat system feels great and the story is solid")


def test_cleaner_drops_short_and_duplicate_paragraphs() -> None:
    cleaner = BoilerplateFilter()
    kept, stats = cleaner.clean(
        [
            "too short",  # < 8 kata -> dibuang
            "The combat is really fun and the story is quite good indeed",  # disimpan
            "The combat is really fun and the story is quite good indeed",  # duplikat
        ]
    )
    assert len(kept) == 1
    assert stats["short"] == 1
    assert stats["duplicate"] == 1


def test_preprocess_document_keeps_order_and_reports_stats(matcher: LexiconMatcher) -> None:
    doc = preprocess_document(
        "Fake Game",
        [
            "The combat of Fake Game is really fun and the story is also good",
            "Copyright 2024 all rights reserved reserved by the publisher company",
            "The graphics are gorgeous and the soundtrack is amazing as well",
        ],
        matcher=matcher,
        budget_tokens=100,
    )
    assert doc.dropped_boilerplate == 1
    assert len(doc.paragraphs) == 2
    # Urutan dokumen harus dipertahankan.
    assert doc.paragraphs[0].startswith("The combat")
    assert doc.paragraphs[1].startswith("The graphics")


def test_paragraph_selector_respects_budget() -> None:
    selector = ParagraphSelector()
    paragraphs = [
        Paragraph(text="ten words exactly here we go", score=0.9, index=0),
        Paragraph(text="another ten words exactly here we go", score=0.8, index=1),
    ]
    selected = selector.select(paragraphs, budget_tokens=25)
    assert len(selected) == 2  # dua paragraf muat di 25 kata
    selected = selector.select(paragraphs, budget_tokens=10)
    assert len(selected) == 1  # hanya yangHighest skor yang muat


def test_preprocess_marks_short_document_invalid(matcher: LexiconMatcher) -> None:
    doc = preprocess_document(
        "Fake Game",
        ["The combat is really fun and quite good"],
        matcher=matcher,
    )
    assert not doc.is_valid


# --------------------------------------------------------------------------- [B] tabel pseudo
def test_pseudo_aspect_table_counts_mentions(matcher: LexiconMatcher) -> None:
    text = (
        "The combat is great. The story is terrible. The combat is fun too. "
        "The graphics are beautiful."
    )
    table = pseudo_aspect_table(text, matcher, top_k=5)
    by_aspect = {e.aspect: e for e in table}
    assert by_aspect[Aspect.GAMEPLAY].freq == 2
    assert by_aspect[Aspect.STORY].freq == 1
    assert by_aspect[Aspect.GRAPHICS].freq == 1
    # Tanpa model sentimen, polaritas default neutral.
    assert by_aspect[Aspect.GAMEPLAY].polarity is Polarity.NEUTRAL


# --------------------------------------------------------------------------- [C] serialisasi
def test_format_aspect_prefix_matches_spec() -> None:
    aspects = [
        AspectEntry(aspect=Aspect.GAMEPLAY, freq=23, polarity=Polarity.POSITIVE),
        AspectEntry(aspect=Aspect.STORY, freq=11, polarity=Polarity.POSITIVE),
        AspectEntry(aspect=Aspect.PERFORMANCE, freq=8, polarity=Polarity.NEGATIVE),
    ]
    prefix = format_aspect_prefix(aspects)
    assert prefix == (
        "aspects: gameplay (23, positive) ; story (11, positive) ; performance (8, negative)"
    )


def test_format_aspect_prefix_respects_top_k() -> None:
    aspects = [
        AspectEntry(aspect=Aspect.GAMEPLAY, freq=23, polarity=Polarity.POSITIVE),
        AspectEntry(aspect=Aspect.STORY, freq=11, polarity=Polarity.POSITIVE),
        AspectEntry(aspect=Aspect.AUDIO, freq=5, polarity=Polarity.NEUTRAL),
    ]
    assert "story" not in format_aspect_prefix(aspects, top_k=1)


def test_serialize_input_joins_prefix_and_body() -> None:
    assert serialize_input("aspects: gameplay (1, positive)", "the combat is fun") == (
        "aspects: gameplay (1, positive) </s> the combat is fun"
    )


def test_build_inference_input_uses_real_aspect_table() -> None:
    inp = _meta_input(
        [_review("The combat is great")],
        [AspectEntry(aspect=Aspect.GAMEPLAY, freq=4, polarity=Polarity.POSITIVE)],
    )
    text = build_inference_input(inp, "body text here")
    assert text.startswith("aspects: gameplay (4, positive) </s> ")


# --------------------------------------------------------------------------- B1 ekstraktif
def test_lead_k_takes_first_sentences() -> None:
    text = "First one. Second one. Third one. Fourth one."
    assert lead_k(text, 2) == "First one. Second one."


def test_textrank_returns_k_sentences() -> None:
    text = (
        "The combat is fun and the story is good. The combat is fun and the story is good. "
        "The graphics are beautiful and the music is great. "
        "The combat is fun and the story is good."
    )
    summary = textrank(text, k=2)
    assert 0 < len(summary.split(".")) <= 3
    assert summary


def test_extractive_generator_satisfies_protocol() -> None:
    inp = _meta_input(
        [_review("The combat is great. The story is bad.")],
        [AspectEntry(aspect=Aspect.GAMEPLAY, freq=4, polarity=Polarity.POSITIVE)],
    )
    generator = ExtractiveMetaReviewGenerator(method="lead", k=1)
    out = generator.generate(inp)
    assert out
    assert "Fake Game" in out


# --------------------------------------------------------------------------- fixture mock
def test_mock_fixtures_satisfy_the_contract() -> None:
    # Fixture mock Task 1 + Task 2 harus valid sebagai MetaReviewInput (payload Task 3).
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    inp = MetaReviewInput.from_json_file(fixtures / "mock_task3_input.json")
    assert inp.game == "Fake Game"
    assert len(inp.reviews) == 12
    assert inp.sentiment.dominant is Polarity.POSITIVE
    assert inp.aspects and inp.aspects[0].aspect is Aspect.GAMEPLAY
    # Fixture output pipeline harus valid sebagai PipelineOutput.
    out = PipelineOutput.from_json_file(fixtures / "mock_pipeline_output.json")
    assert out.game == inp.game
    assert out.meta_review
    assert [e.aspect for e in out.aspects] == [e.aspect for e in inp.aspects]


# --------------------------------------------------------------------------- metrik
def test_rouge_scores_reward_overlap() -> None:
    identical = rouge_scores("the combat is fun", "the combat is fun")
    different = rouge_scores("the combat is fun", "completely unrelated words here")
    assert identical["rouge1"] == pytest.approx(1.0)
    assert different["rouge1"] == pytest.approx(0.0)


def test_average_rouge_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        average_rouge(["a"], ["a", "b"])


def test_aspect_coverage_counts_mentioned_aspects(matcher: LexiconMatcher) -> None:
    aspects = [
        AspectEntry(aspect=Aspect.GAMEPLAY, freq=4, polarity=Polarity.POSITIVE),
        AspectEntry(aspect=Aspect.STORY, freq=2, polarity=Polarity.NEGATIVE),
    ]
    coverage = aspect_coverage(aspects, "The combat system is great.", matcher)
    assert coverage == pytest.approx(0.5)


def test_polarity_consistency_agrees_with_table(matcher: LexiconMatcher) -> None:
    aspects = [AspectEntry(aspect=Aspect.GAMEPLAY, freq=4, polarity=Polarity.POSITIVE)]
    good = polarity_consistency(aspects, "The combat is amazing and fun.", matcher)
    bad = polarity_consistency(aspects, "The combat is awful and boring.", matcher)
    assert good == pytest.approx(1.0)
    assert bad == pytest.approx(0.0)


def test_paired_bootstrap_reports_difference() -> None:
    # Sistem identik -> selisih nol, p-value besar.
    same = paired_bootstrap([0.5] * 20, [0.5] * 20, n_resamples=200, seed=1)
    assert same["mean_diff"] == pytest.approx(0.0)
    assert 0.0 < same["p_value"] <= 1.0
    # Sistem A selalu lebih baik -> selisih positif dan interval keyakinan di atas nol.
    diff = paired_bootstrap([1.0] * 20, [0.0] * 20, n_resamples=200, seed=1)
    assert diff["mean_diff"] == pytest.approx(1.0)
    assert diff["ci_low"] > 0.0
