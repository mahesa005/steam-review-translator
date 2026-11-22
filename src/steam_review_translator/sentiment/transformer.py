"""S2-S4: Hugging Face sequence classifiers (needs the ``task1`` extra: transformers, torch).

- S2: a pretrained 3-class sentiment model used zero-shot (default
  ``cardiffnlp/twitter-roberta-base-sentiment-latest``).
- S3/S4: BERT-base / RoBERTa-base fine-tuned on Metacritic by
  ``experiments/task1_sentiment/train_transformer.py``; point ``model`` at the saved dir.

The model's ``id2label`` is mapped onto :class:`Polarity` by name (``positive``/``pos``/
``LABEL_2`` ...), so any 3-class sentiment head works as long as its labels are named.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from steam_review_translator.core.schemas import Polarity
from steam_review_translator.sentiment.base import ReviewClassifier, Scores

_LABEL_ALIASES: dict[str, Polarity] = {
    "positive": Polarity.POSITIVE,
    "pos": Polarity.POSITIVE,
    "negative": Polarity.NEGATIVE,
    "neg": Polarity.NEGATIVE,
    "neutral": Polarity.NEUTRAL,
    "neu": Polarity.NEUTRAL,
    "mixed": Polarity.NEUTRAL,
}


def map_label(label: str) -> Polarity:
    """Map a model label name to a polarity (case-insensitive)."""
    key = label.strip().lower()
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    raise ValueError(
        f"cannot map model label {label!r} to a polarity; give the model named labels "
        "(positive/negative/neutral) in its config.id2label"
    )


def pick_device() -> str:
    """cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformerSentimentAnalyzer(ReviewClassifier):
    """Batched softmax scores from any HF sequence-classification checkpoint."""

    def __init__(
        self,
        model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
        name: str | None = None,
    ) -> None:
        """Load tokenizer and model; ``max_length`` truncates long reviews."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name or f"hf:{model}"
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or pick_device()
        self._torch = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model)
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.to(self.device).eval()
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self._labels = [map_label(id2label[i]) for i in sorted(id2label)]

    def scores(self, texts: Sequence[str]) -> list[Scores]:
        """Softmax probabilities per class."""
        torch = self._torch
        out: list[Scores] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            for row in probs:
                scores: Scores = dict.fromkeys(Polarity, 0.0)
                for label, p in zip(self._labels, row, strict=True):
                    scores[label] += float(p)
                out.append(scores)
        return out
