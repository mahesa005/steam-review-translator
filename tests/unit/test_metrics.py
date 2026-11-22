import pytest

from steam_review_translator.aspect.metrics import score
from steam_review_translator.core.schemas import Aspect, AspectLabel, Polarity

POS, NEG = Polarity.POSITIVE, Polarity.NEGATIVE


def L(aspect: Aspect, polarity: Polarity) -> AspectLabel:
    return AspectLabel(aspect=aspect, polarity=polarity)


def test_perfect_prediction() -> None:
    gold = [[L(Aspect.STORY, POS)], [], [L(Aspect.GAMEPLAY, NEG), L(Aspect.OVERALL, NEG)]]
    s = score(gold, gold)
    assert s["pair"]["f1"] == 1.0
    assert s["exact"] == 1.0
    assert s["polarity_acc"] == 1.0


def test_wrong_polarity_counts_for_aspect_not_pair() -> None:
    s = score([[L(Aspect.STORY, POS)]], [[L(Aspect.STORY, NEG)]])
    assert s["pair"]["f1"] == 0.0
    assert s["aspect"]["f1"] == 1.0
    assert s["polarity_acc"] == 0.0
    assert s["per_aspect"]["story"] == {"p": 1.0, "r": 1.0, "f1": 1.0, "support": 1}


def test_missing_and_extra() -> None:
    gold = [[L(Aspect.STORY, POS), L(Aspect.AUDIO, POS)]]
    pred = [[L(Aspect.STORY, POS), L(Aspect.PRICE, NEG)]]
    s = score(gold, pred)
    assert s["pair"]["p"] == 0.5 and s["pair"]["r"] == 0.5
    assert s["per_aspect"]["audio"]["r"] == 0.0
    assert s["per_aspect"]["price"]["p"] == 0.0
    assert s["per_aspect"]["price"]["support"] == 0
    assert s["exact"] == 0.0


def test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        score([[]], [[], []])
