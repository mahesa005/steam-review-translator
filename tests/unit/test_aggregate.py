from steam_review_translator.aspect.aggregate import aggregate
from steam_review_translator.core.schemas import Aspect, AspectEntry, AspectLabel, Polarity

POS, NEG, NEU = Polarity.POSITIVE, Polarity.NEGATIVE, Polarity.NEUTRAL


def L(aspect: Aspect, polarity: Polarity) -> AspectLabel:
    return AspectLabel(aspect=aspect, polarity=polarity)


def test_empty_input() -> None:
    assert aggregate([]) == []
    assert aggregate([[], [], []]) == []


def test_single_aspect_majority() -> None:
    labels = [[L(Aspect.STORY, POS)], [L(Aspect.STORY, POS)], [L(Aspect.STORY, NEG)]]
    assert aggregate(labels) == [AspectEntry(aspect=Aspect.STORY, freq=3, polarity=POS)]


def test_tie_is_neutral() -> None:
    labels = [[L(Aspect.PRICE, POS)], [L(Aspect.PRICE, NEG)]]
    assert aggregate(labels)[0].polarity is NEU

    three_way = [[L(Aspect.PRICE, POS)], [L(Aspect.PRICE, NEG)], [L(Aspect.PRICE, NEU)]]
    assert aggregate(three_way)[0].polarity is NEU


def test_tie_below_top_does_not_matter() -> None:
    labels = [[L(Aspect.AUDIO, POS)]] * 3 + [[L(Aspect.AUDIO, NEG)], [L(Aspect.AUDIO, NEU)]]
    assert aggregate(labels)[0].polarity is POS


def test_multiple_aspects_sorted_by_freq_then_name() -> None:
    labels = [
        [L(Aspect.GAMEPLAY, POS), L(Aspect.PERFORMANCE, NEG)],
        [L(Aspect.PERFORMANCE, NEG)],
        [L(Aspect.PERFORMANCE, NEG), L(Aspect.AUDIO, POS)],
        [L(Aspect.GAMEPLAY, POS)],
        [],
    ]
    assert aggregate(labels) == [
        AspectEntry(aspect=Aspect.PERFORMANCE, freq=3, polarity=NEG),
        AspectEntry(aspect=Aspect.GAMEPLAY, freq=2, polarity=POS),
        AspectEntry(aspect=Aspect.AUDIO, freq=1, polarity=POS),
    ]


def test_same_aspect_counted_once_per_sentence() -> None:
    labels = [[L(Aspect.STORY, POS), L(Aspect.STORY, NEG)], [L(Aspect.STORY, POS)]]
    assert aggregate(labels) == [AspectEntry(aspect=Aspect.STORY, freq=2, polarity=POS)]
