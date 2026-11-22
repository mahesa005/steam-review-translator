"""Generator meta-review seq2seq (BART / T5) untuk Task 3 (owner: Lukas).

Mengimplementasikan skenario:

- **B2** ``BART-large-cnn`` zero-shot  -> ``BartMetaReviewGenerator`` tanpa fine-tuning.
- **E1** ``BART-base`` fine-tuned, input teks saja        -> ``use_aspect_prefix=False``.
- **E2** ``BART-base`` fine-tuned + prefix aspek          -> ``use_aspect_prefix=True``.
- **E3** ``T5-base``   fine-tuned + prefix aspek          -> ``T5MetaReviewGenerator``.

Alur ``generate``:
    MetaReviewInput -> praproses [A] -> prefix aspek [C] -> encode -> beam search -> decode

Library berat (torch, transformers) di-import di dalam fungsi supaya paket inti tetap
ringan dan modul ini hanya termuat bila implementasinya benar-benar dipilih.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.logging import get_logger
from steam_review_translator.core.schemas import MetaReviewInput
from steam_review_translator.summarization.preprocessing import preprocess_reviews
from steam_review_translator.summarization.serialization import build_inference_input

log = get_logger(__name__)

#: Model default per arsitektur.
DEFAULT_BART = "facebook/bart-base"
DEFAULT_BART_CNN = "facebook/bart-large-cnn"
DEFAULT_T5 = "t5-base"

#: Batas input per model (spec: Max input / target 1.024 / 256 untuk BART, 512 / 256 T5).
MAX_INPUT_TOKENS = {"bart": 1024, "t5": 512}

#: Budget token untuk seleksi paragraf (spec [A] langkah 4).
BUDGET_TOKENS = {"bart": 800, "t5": 400}

#: Batas token target (spec: 256 token).
MAX_NEW_TOKENS = 256

_WS_RE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    """Ratakan teks menjadi satu paragraf."""
    return _WS_RE.sub(" ", text).strip()


class TransformerMetaReviewGenerator:
    """Generator meta-review berbasis seq2seq (BART atau T5).

    Parameter mengikuti spec bagian 4 (Hyperparameter awal) dan bagian 3 [A]/[C].
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BART,
        *,
        architecture: str = "bart",
        checkpoint: str | Path | None = None,
        use_aspect_prefix: bool = True,
        top_k_aspects: int = 5,
        max_input_tokens: int | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        budget_tokens: int | None = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        min_new_tokens: int = 20,
        device: str | None = None,
        lexicon: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        """Konfigurasi model & decoding.

        - ``architecture``: ``"bart"`` atau ``"t5"`` (menentukan batas token & pemisah).
        - ``checkpoint``: path direktori model hasil fine-tuning (menimpa ``model_name``).
        - ``use_aspect_prefix``: False -> E1 (teks saja), True -> E2/E3 (prefix aspek).
        - ``seed``: buat reproducible (digunakan eksperimen, bukan saat generate).
        """
        if architecture not in {"bart", "t5"}:
            raise ValueError(f"unknown architecture {architecture!r}; expected 'bart' or 't5'")
        self.architecture = architecture
        self.model_name = str(checkpoint) if checkpoint else model_name
        self.use_aspect_prefix = use_aspect_prefix
        self.top_k_aspects = top_k_aspects
        self.max_input_tokens = max_input_tokens or MAX_INPUT_TOKENS[architecture]
        self.max_new_tokens = max_new_tokens
        self.budget_tokens = budget_tokens or BUDGET_TOKENS[architecture]
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.min_new_tokens = min_new_tokens
        self.device = device
        self.seed = seed
        self._matcher = LexiconMatcher(load_lexicon(lexicon))
        self._tokenizer: Any = None
        self._model: Any = None

    # ------------------------------------------------------------------ model loading
    def _resolve_device(self) -> str:
        """Pilih device: param eksplisit > cuda > cpu."""
        if self.device:
            return self.device
        import torch  # import lokal: library berat

        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Muat tokenizer + model (sekali saja, lalu di-cache)."""
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # import lokal

        device = self._resolve_device()
        log.info("loading %s on %s", self.model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(device)
        model.eval()
        self._model = model

    def _token_counter(self) -> Callable[[str], int]:
        """Fungsi penghitung token; pakai tokenizer bila model sudah dimuat."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer  # import lokal

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._tokenizer = tokenizer

        def count(text: str) -> int:
            return len(tokenizer(text)["input_ids"])

        return count

    # ------------------------------------------------------------------ input building
    def build_body(self, inp: MetaReviewInput) -> str:
        """Tahap [A]: pilih paragraf terpilih dari review yang masuk."""
        texts = [r.text for r in inp.reviews]
        doc = preprocess_reviews(
            inp.game,
            texts,
            matcher=self._matcher,
            budget_tokens=self.budget_tokens,
            token_counter=self._token_counter(),
        )
        return doc.text

    def build_input_text(self, inp: MetaReviewInput) -> str:
        """Tahap [C]: rangkai prefix aspek + teks terpilih jadi satu string input."""
        body = self.build_body(inp)
        if not self.use_aspect_prefix:
            return body  # E1: teks saja
        return build_inference_input(inp, body, top_k=self.top_k_aspects)

    # ------------------------------------------------------------------ generation
    def generate(self, inp: MetaReviewInput) -> str:
        """Tulis satu paragraf meta-review untuk ``inp``."""
        text = self.generate_text(self.build_input_text(inp))
        if not text.strip():
            # Jangan pernah kembalikan string kosong (Protocol mensyaratkan non-empty).
            return f"No review text available for {inp.game}."
        return _collapse(text)

    def generate_text(self, input_text: str) -> str:
        """Encode -> beam search -> decode untuk satu string input."""
        self.load()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:  # pragma: no cover - dijamin oleh load()
            raise RuntimeError("model gagal dimuat")
        device = self._resolve_device()
        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(device)
        with torch_no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                early_stopping=True,
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_batch(self, inputs: Sequence[str], batch_size: int = 8) -> list[str]:
        """Generate untuk banyak string input (dipakai di evaluasi)."""
        self.load()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:  # pragma: no cover - dijamin oleh load()
            raise RuntimeError("model gagal dimuat")
        device = self._resolve_device()
        results: list[str] = []
        for start in range(0, len(inputs), batch_size):
            chunk = list(inputs[start : start + batch_size])
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(device)
            with torch_no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    early_stopping=True,
                )
            results.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        return results


def torch_no_grad() -> Any:
    """``torch.no_grad()`` tanpa meng-import torch di level modul."""
    import torch  # import lokal

    return torch.no_grad()


class BartMetaReviewGenerator(TransformerMetaReviewGenerator):
    """Generator BART (B2 zero-shot, E1 tanpa prefix, E2 dengan prefix)."""

    def __init__(
        self,
        model_name: str = DEFAULT_BART,
        **kwargs: Any,
    ) -> None:
        """Semua parameter lain sama dengan :class:`TransformerMetaReviewGenerator`."""
        kwargs.setdefault("architecture", "bart")
        super().__init__(model_name, **kwargs)


class T5MetaReviewGenerator(TransformerMetaReviewGenerator):
    """Generator T5 (E3: T5-base fine-tuned + prefix aspek)."""

    def __init__(
        self,
        model_name: str = DEFAULT_T5,
        **kwargs: Any,
    ) -> None:
        """Semua parameter lain sama dengan :class:`TransformerMetaReviewGenerator`."""
        kwargs.setdefault("architecture", "t5")
        super().__init__(model_name, **kwargs)
