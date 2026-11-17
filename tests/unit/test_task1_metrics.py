import math
from collections.abc import Sequence

import pytest

from steam_review_translator.core.interfaces import SentimentAnalyzer
from steam_review_translator.core.schemas import Polarity, Review
from steam_review_translator.sentiment.base import ReviewClassifier, Scores
from steam_review_translator.sentiment.labels import (
    BINARY_CLASSES,
    CLASSES,
    collapse_to_binary,
    score_to_polarity,
    shares,
)
from steam_review_translator.sentiment.metrics import (
    adjusted_classify_and_count,
    classification_report,
    confusion_matrix,
    distribution_report,
    mae,
    misclassification_rates,
    pearson,
    total_variation,
)

POS, NEG, NEU = Polarity.POSITIVE, Polarity.NEGATIVE, Polarity.NEUTRAL


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0, NEG), (4, NEG), (4.5, NEU), (5, NEU), (7, NEU), (7.9, NEU), (8, POS), (10, POS)],
)
def test_score_to_polarity_default_bands(score: float, expected: Polarity) -> None:
    assert score_to_polarity(score) is expected


def test_score_to_polarity_custom_and_invalid() -> None:
    assert score_to_polarity(7, positive_min=7) is POS
    with pytest.raises(ValueError):
        score_to_polarity(11)
    with pytest.raises(ValueError):
        score_to_polarity(5, negative_max=6, positive_min=6)


def test_collapse_to_binary_ignores_neutral() -> None:
    assert collapse_to_binary({POS: 0.2, NEG: 0.1, NEU: 0.7}) is POS
    assert collapse_to_binary({POS: 0.1, NEG: 0.3, NEU: 0.6}) is NEG


def test_shares() -> None:
    assert shares([POS, POS, NEG, NEU]) == [0.5, 0.25, 0.25]
    assert shares([], BINARY_CLASSES) == [0.0, 0.0]


def test_classification_report() -> None:
    gold = [POS, POS, NEG, NEU]
    pred = [POS, NEG, NEG, NEU]
    rep = classification_report(gold, pred, CLASSES)
    assert rep["accuracy"] == 0.75
    assert rep["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert rep["per_class"]["positive"]["recall"] == 0.5
    assert rep["per_class"]["negative"]["precision"] == 0.5
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_confusion_matrix_skips_out_of_scheme_labels() -> None:
    assert confusion_matrix([POS, NEU], [POS, POS], BINARY_CLASSES) == [[1, 0], [0, 0]]
    with pytest.raises(ValueError):
        confusion_matrix([POS], [], CLASSES)


def test_mae_and_tvd() -> None:
    gold, pred = [0.6, 0.3, 0.1], [0.5, 0.3, 0.2]
    assert mae(gold, pred) == pytest.approx(0.2 / 3)
    assert total_variation(gold, pred) == pytest.approx(0.1)


def test_distribution_report_bias_and_dominant() -> None:
    gold = [[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]]
    pred = [[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]]
    rep = distribution_report(gold, pred, BINARY_CLASSES)
    assert rep["n_games"] == 3
    assert rep["dominant_agreement"] == pytest.approx(2 / 3)
    assert rep["bias"]["positive"] == pytest.approx(0.1)
    assert rep["bias"]["negative"] == pytest.approx(-0.1)
    assert rep["pearson_positive"] == pytest.approx(pearson([0.7, 0.4, 0.9], [0.8, 0.6, 0.9]))


def test_pearson_degenerate() -> None:
    assert math.isnan(pearson([1.0, 1.0], [0.0, 1.0]))
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_acc_recovers_true_prevalence() -> None:
    # Classifier calls 20% of positives negative and 10% of negatives positive.
    rates = misclassification_rates([[80, 20], [10, 90]])
    true = [0.7, 0.3]
    observed = [true[0] * 0.8 + true[1] * 0.1, true[0] * 0.2 + true[1] * 0.9]
    assert adjusted_classify_and_count(observed, rates) == pytest.approx(true)


def test_acc_clips_and_handles_singular() -> None:
    rates = misclassification_rates([[80, 20], [10, 90]])
    corrected = adjusted_classify_and_count([1.0, 0.0], rates)
    assert corrected == pytest.approx([1.0, 0.0])
    singular = [[0.5, 0.5], [0.5, 0.5]]
    assert adjusted_classify_and_count([0.6, 0.4], singular) == [0.6, 0.4]
    assert misclassification_rates([[0, 0], [1, 1]])[0] == [1.0, 0.0]


class _Fixed(ReviewClassifier):
    def scores(self, texts: Sequence[str]) -> list[Scores]:
        table = {"good": {POS: 0.8, NEG: 0.1, NEU: 0.1}, "meh": {POS: 0.3, NEG: 0.2, NEU: 0.5}}
        return [table.get(t, {POS: 0.1, NEG: 0.8, NEU: 0.1}) for t in texts]


def test_review_classifier_is_a_sentiment_analyzer() -> None:
    clf = _Fixed()
    assert isinstance(clf, SentimentAnalyzer)
    reviews = [
        Review(review_id="1", text="good", recommended=False),  # recommended must be ignored
        Review(review_id="2", text="meh"),
        Review(review_id="3", text="bad"),
        Review(review_id="4", text="good"),
    ]
    dist = clf.analyze(reviews)
    assert (dist.positive, dist.negative, dist.neutral) == (0.5, 0.25, 0.25)
    assert dist.dominant is POS
    assert clf.predict_binary(["meh"]) == [POS]
    assert clf.analyze([]).neutral == 1.0
