"""S1: TF-IDF + logistic regression (needs the ``task1`` extra: scikit-learn).

Train with ``experiments/task1_sentiment/train_tfidf.py``; the pipeline loads the saved
model via ``model_path``.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from steam_review_translator.core.config import resolve_path
from steam_review_translator.core.schemas import Polarity
from steam_review_translator.sentiment.base import ReviewClassifier, Scores


class TfidfSentimentAnalyzer(ReviewClassifier):
    """Wraps a fitted sklearn ``Pipeline(TfidfVectorizer, LogisticRegression)``."""

    name = "S1-tfidf-lr"

    def __init__(self, model_path: str | Path | None = None, model: Any = None) -> None:
        """Load a pickled sklearn pipeline from ``model_path`` or take a fitted ``model``."""
        if model is None:
            if model_path is None:
                raise ValueError("give model_path or model")
            with resolve_path(model_path).open("rb") as fh:
                model = pickle.load(fh)  # noqa: S301 - our own artifact
        self.model = model
        self._classes = [Polarity(c) for c in model.classes_]

    def scores(self, texts: Sequence[str]) -> list[Scores]:
        """Class probabilities from the logistic regression."""
        probs = self.model.predict_proba(list(texts))
        return [{c: float(p) for c, p in zip(self._classes, row, strict=True)} for row in probs]

    def save(self, path: str | Path) -> Path:
        """Pickle the fitted pipeline."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(self.model, fh)
        return out
