"""System C: a transformer encoder fine-tuned with LoRA (PEFT) on the LLM-labeled sentences.

One encoder pass per sentence, mean-pooled, then one 4-way head per aspect:
``none | positive | negative | neutral``. An aspect is predicted when ``1 - P(none)`` reaches
the threshold tuned on val; its polarity is the most likely of the three polarity classes.

Artifact layout (``model_dir``, written by experiments/task2_aspect/systems/train_peft.py):
    adapter/   LoRA weights (peft ``save_pretrained``)
    head.pt    state dict of the classification head
    meta.json  {"base_model", "max_length", "threshold", ...}

torch, transformers and peft are imported lazily, only when this tagger is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from steam_review_translator.core.config import project_root
from steam_review_translator.core.schemas import Aspect, AspectLabel, Polarity

ASPECTS: list[Aspect] = list(Aspect)
CLASSES: list[Polarity | None] = [None, Polarity.POSITIVE, Polarity.NEGATIVE, Polarity.NEUTRAL]


def encode_labels(labels: list[AspectLabel]) -> list[int]:
    """Class index per aspect (0 = not mentioned) for one sentence."""
    by_aspect = {x.aspect: x.polarity for x in labels}
    return [CLASSES.index(by_aspect.get(a)) for a in ASPECTS]


def build_model(base_model: str) -> Any:
    """Encoder + per-aspect head (``torch.nn.Module``); the encoder is not yet LoRA-wrapped."""
    import torch
    from transformers import AutoModel

    class AspectModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_model)
            hidden = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(hidden, len(ASPECTS) * len(CLASSES))

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            logits = self.head(self.dropout(pooled))
            return logits.view(-1, len(ASPECTS), len(CLASSES))

    return AspectModel()


def decode(probs: Any, threshold: float) -> list[list[AspectLabel]]:
    """Labels from class probabilities of shape (n, aspects, classes)."""
    out: list[list[AspectLabel]] = []
    for row in probs.tolist():
        labels = []
        for aspect, p in zip(ASPECTS, row, strict=True):
            if 1.0 - p[0] >= threshold:
                k = max(range(1, len(CLASSES)), key=lambda i: p[i])
                labels.append(AspectLabel(aspect=aspect, polarity=CLASSES[k]))
        out.append(labels)
    return out


class PeftAspectTagger:
    """Load a trained LoRA aspect model and tag sentences."""

    def __init__(
        self, model_dir: str | Path = "artifacts/task2/peft", batch_size: int | str = 32
    ) -> None:
        """Load base encoder + LoRA adapter + head from ``model_dir``."""
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer

        d = Path(model_dir)
        if not d.is_absolute():
            d = project_root() / d
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.batch_size = int(batch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.meta["base_model"])
        model = build_model(self.meta["base_model"])
        model.encoder = PeftModel.from_pretrained(model.encoder, d / "adapter").merge_and_unload()
        model.head.load_state_dict(torch.load(d / "head.pt", map_location="cpu"))
        self.model = model.to(self.device).eval()

    def probs(self, sentences: list[str]) -> Any:
        """Class probabilities, shape (n, aspects, classes), on the CPU."""
        import torch

        chunks = []
        with torch.no_grad():
            for i in range(0, len(sentences), self.batch_size):
                enc = self.tokenizer(
                    sentences[i : i + self.batch_size],
                    truncation=True,
                    max_length=self.meta["max_length"],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                chunks.append(logits.softmax(-1).cpu())
        if not chunks:
            return torch.zeros(0, len(ASPECTS), len(CLASSES))
        return torch.cat(chunks)

    def tag(self, sentences: list[str]) -> list[list[AspectLabel]]:
        """Return one list of labels per sentence."""
        return decode(self.probs(sentences), self.meta["threshold"])
