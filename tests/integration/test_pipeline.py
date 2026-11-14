from pathlib import Path

import pytest

from steam_review_translator.cli import main
from steam_review_translator.core.config import (
    ComponentConfig,
    PipelineConfig,
    load_pipeline_config,
)
from steam_review_translator.core.schemas import (
    AspectLabel,
    PipelineInput,
    PipelineOutput,
    Polarity,
    reviews_from_json_file,
)
from steam_review_translator.pipeline import Pipeline, build_pipeline

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_reviews.json"


@pytest.fixture(scope="module")
def output() -> PipelineOutput:
    pipeline = build_pipeline(load_pipeline_config())
    reviews = reviews_from_json_file(FIXTURE)
    return pipeline.run(PipelineInput(game="Fake Game", reviews=reviews))


def test_fixture_shape() -> None:
    assert len(reviews_from_json_file(FIXTURE)) == 12


def test_output_validates(output: PipelineOutput) -> None:
    again = PipelineOutput.model_validate_json(output.model_dump_json())
    assert again == output
    assert output.game == "Fake Game"


def test_aspects_sorted_by_freq(output: PipelineOutput) -> None:
    freqs = [e.freq for e in output.aspects]
    assert freqs == sorted(freqs, reverse=True)
    assert len(output.aspects) >= 5  # fixture covers many aspects


def test_meta_review_non_empty(output: PipelineOutput) -> None:
    assert output.meta_review.strip()
    assert "Fake Game" in output.meta_review


def test_mock_results_are_deterministic(output: PipelineOutput) -> None:
    # 5 recommended, 4 not, 3 unflagged: r012 is lexically positive, r006/r010 have no cues.
    s = output.sentiment
    assert (s.positive, s.negative, s.neutral) == pytest.approx((6 / 12, 4 / 12, 2 / 12))
    assert s.dominant is Polarity.POSITIVE


def test_unknown_impl_rejected() -> None:
    cfg = PipelineConfig(sentiment=ComponentConfig(impl="does-not-exist"))
    with pytest.raises(ValueError, match="registered"):
        build_pipeline(cfg)


def test_tagger_contract_enforced() -> None:
    class BadTagger:
        def tag(self, sentences: list[str]) -> list[list[AspectLabel]]:
            return []

    good = build_pipeline()
    pipeline = Pipeline(good.sentiment, BadTagger(), good.generator)
    reviews = reviews_from_json_file(FIXTURE)
    with pytest.raises(ValueError, match="one per sentence"):
        pipeline.run(PipelineInput(game="G", reviews=reviews))


def test_cli_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "result.json"
    code = main(["demo", "--reviews", str(FIXTURE), "--game", "Fake Game", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    for section in ("[Task 1]", "[Task 2]", "[Task 3]"):
        assert section in printed
    assert PipelineOutput.from_json_file(out).game == "Fake Game"


def test_cli_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--reviews", "nope.json", "--game", "G"]) == 1
    assert "error:" in capsys.readouterr().err
