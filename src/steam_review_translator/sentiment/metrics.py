"""Task 1 evaluation: review-level classification and game-level distribution metrics.

Pure Python (no numpy/sklearn) so it runs in CI with only the core install.

Distribution vectors are lists of class shares in a fixed class order (see
``labels.CLASSES`` / ``labels.BINARY_CLASSES``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from steam_review_translator.core.schemas import Polarity

Matrix = list[list[float]]

# --------------------------------------------------------------------------------------
# Review level
# --------------------------------------------------------------------------------------


def confusion_matrix(
    gold: Sequence[Polarity], pred: Sequence[Polarity], classes: Sequence[Polarity]
) -> list[list[int]]:
    """Counts ``m[i][j]`` = number of items with gold ``classes[i]`` predicted ``classes[j]``.

    Items whose gold or predicted label is outside ``classes`` are skipped.
    """
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} items, pred has {len(pred)}")
    index = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for g, p in zip(gold, pred, strict=True):
        if g in index and p in index:
            m[index[g]][index[p]] += 1
    return m


def classification_report(
    gold: Sequence[Polarity], pred: Sequence[Polarity], classes: Sequence[Polarity]
) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class precision/recall/F1/support, and the confusion matrix."""
    m = confusion_matrix(gold, pred, classes)
    total = sum(map(sum, m))
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for i, c in enumerate(classes):
        tp = m[i][i]
        support = sum(m[i])
        predicted = sum(row[i] for row in m)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
        f1s.append(f1)
    return {
        "n": total,
        "accuracy": sum(m[i][i] for i in range(len(classes))) / total if total else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "classes": [c.value for c in classes],
        "confusion_matrix": m,
    }


# --------------------------------------------------------------------------------------
# Distribution level (one gold/pred share vector per game)
# --------------------------------------------------------------------------------------


def mae(gold: Sequence[float], pred: Sequence[float]) -> float:
    """Mean absolute error across classes."""
    _check_same_len(gold, pred)
    return sum(abs(g - p) for g, p in zip(gold, pred, strict=True)) / len(gold)


def total_variation(gold: Sequence[float], pred: Sequence[float]) -> float:
    """Total variation distance: half the L1 distance (in [0, 1])."""
    _check_same_len(gold, pred)
    return 0.5 * sum(abs(g - p) for g, p in zip(gold, pred, strict=True))


def dominant_index(dist: Sequence[float], tie_index: int | None = None) -> int:
    """Index of the largest share. Ties go to ``tie_index`` if given, else the first max."""
    best = max(dist)
    maxima = [i for i, v in enumerate(dist) if math.isclose(v, best, abs_tol=1e-9)]
    if len(maxima) > 1 and tie_index is not None:
        return tie_index
    return maxima[0]


def distribution_report(
    gold: Sequence[Sequence[float]],
    pred: Sequence[Sequence[float]],
    classes: Sequence[Polarity],
) -> dict[str, Any]:
    """Aggregate distribution metrics over games.

    - ``mae`` / ``tvd``: mean over games.
    - ``dominant_agreement``: share of games whose predicted dominant class matches gold.
    - ``bias``: mean signed error (pred - gold) per class; positive = over-predicted.
    - ``pearson_<class>``: correlation of per-game shares across games (NaN if constant).
    """
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} games, pred has {len(pred)}")
    n = len(gold)
    if n == 0:
        raise ValueError("no games to evaluate")
    tie = classes.index(Polarity.NEUTRAL) if Polarity.NEUTRAL in classes else None
    report: dict[str, Any] = {
        "n_games": n,
        "mae": sum(mae(g, p) for g, p in zip(gold, pred, strict=True)) / n,
        "tvd": sum(total_variation(g, p) for g, p in zip(gold, pred, strict=True)) / n,
        "dominant_agreement": sum(
            dominant_index(g, tie) == dominant_index(p, tie)
            for g, p in zip(gold, pred, strict=True)
        )
        / n,
        "bias": {
            c.value: sum(p[i] - g[i] for g, p in zip(gold, pred, strict=True)) / n
            for i, c in enumerate(classes)
        },
    }
    for i, c in enumerate(classes):
        report[f"pearson_{c.value}"] = pearson([g[i] for g in gold], [p[i] for p in pred])
    return report


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; NaN when either side has zero variance or fewer than 2 points."""
    _check_same_len(x, y)
    n = len(x)
    if n < 2:
        return math.nan
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return math.nan
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / math.sqrt(sxx * syy)


# --------------------------------------------------------------------------------------
# Adjusted classify-and-count (ACC)
# --------------------------------------------------------------------------------------


def misclassification_rates(counts: Sequence[Sequence[int]]) -> Matrix:
    """Row-normalise a confusion matrix into ``P(pred=j | gold=i)``.

    A class with no gold support gets an identity row (assume it is predicted perfectly).
    """
    k = len(counts)
    rates: Matrix = []
    for i, row in enumerate(counts):
        total = sum(row)
        if total == 0:
            rates.append([1.0 if j == i else 0.0 for j in range(k)])
        else:
            rates.append([v / total for v in row])
    return rates


def adjusted_classify_and_count(observed: Sequence[float], rates: Matrix) -> list[float]:
    """Correct observed predicted shares with ACC (Forman 2008).

    Solves ``observed = rates^T @ true`` for ``true``, clips negatives to 0, and
    renormalises to sum 1. If the system is singular, returns ``observed`` unchanged.
    """
    k = len(observed)
    if len(rates) != k or any(len(r) != k for r in rates):
        raise ValueError("rates must be a k x k matrix matching observed")
    transposed = [[rates[j][i] for j in range(k)] for i in range(k)]
    solved = _solve(transposed, list(observed))
    if solved is None:
        return list(observed)
    clipped = [max(0.0, v) for v in solved]
    total = sum(clipped)
    if total == 0:
        return list(observed)
    return [v / total for v in clipped]


def _solve(a: Matrix, b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. Returns None if (near-)singular."""
    k = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(k):
            if r != col:
                factor = aug[r][col] / aug[col][col]
                for c in range(col, k + 1):
                    aug[r][c] -= factor * aug[col][c]
    return [aug[i][k] / aug[i][i] for i in range(k)]


def _check_same_len(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b) or not a:
        raise ValueError(f"expected two non-empty vectors of equal length, got {len(a)}, {len(b)}")
