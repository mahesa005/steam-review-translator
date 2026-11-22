"""Skenario B1 Task 3: baseline ekstraktif Lead-k dan TextRank (owner: Lukas).

B1 adalah "baseline bawah": tanpa training, tanpa model. Dipakai sebagai titik
pembanding untuk RQ1 (model seq2seq fine-tuned vs baseline ekstraktif).

- **Lead-k**   : ambil ``k`` kalimat pertama (ulasan Steam biasanya memuat opini
                 utama di awal).
- **TextRank** :.PageRank pada graf kemiripan kalimat (implementasi pure Python,
                 tanpa networkx).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from steam_review_translator.aspect.sentences import split_sentences
from steam_review_translator.core.logging import get_logger
from steam_review_translator.core.schemas import MetaReviewInput

log = get_logger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "at",
        "by",
        "for",
        "with",
        "to",
        "from",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "my",
        "your",
        "his",
        "her",
        "their",
        "our",
        "as",
        "so",
        "very",
        "just",
    }
)


def tokenize_words(text: str) -> list[str]:
    """Tokenisasi kata sederhana (huruf kecil, alphanumeric)."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _content_tokens(text: str) -> set[str]:
    """Set token isi (tanpa stopword) untuk menghitung kemiripan kalimat."""
    return {t for t in tokenize_words(text) if t not in _STOPWORDS}


def lead_k(text: str, k: int = 3) -> str:
    """Baseline Lead-k: ``k`` kalimat pertama dari ``text``."""
    sentences = split_sentences(text)
    return " ".join(sentences[:k]).strip()


def _similarity_matrix(sentences: Sequence[str]) -> list[list[float]]:
    """Matriks kemiripan kosinus antar kalimat (based on token isi)."""
    vectors = [_content_tokens(s) for s in sentences]
    n = len(sentences)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vectors[i], vectors[j]
            if not a or not b:
                continue
            dot = len(a & b)
            if dot == 0:
                continue
            denom = math.sqrt(len(a)) * math.sqrt(len(b))
            sim = dot / denom
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def textrank(text: str, k: int = 3, damping: float = 0.85, iterations: int = 30) -> str:
    """Baseline TextRank: ``k`` kalimat dengan skor PageRank tertinggi.

    Implementasi PageRank pure Python; graf tak berarah dengan bobot kemiripan
    kosinus antar kalimat.
    """
    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return ""
    if n <= k:
        return " ".join(sentences).strip()
    matrix = _similarity_matrix(sentences)
    # Normalisasi baris: probabilitas keluar tiap simpul.
    out_sum = [sum(row) or 1.0 for row in matrix]
    scores = [1.0 / n] * n
    for _ in range(iterations):
        new_scores = []
        for i in range(n):
            acc = (1.0 - damping) / n
            for j in range(n):
                if matrix[j][i] > 0:
                    acc += damping * scores[j] * matrix[j][i] / out_sum[j]
            new_scores.append(acc)
        total = sum(new_scores) or 1.0
        scores = [s / total for s in new_scores]
    ranked = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
    # Kembalikan kalimat dalam urutan kemunculannya (lebih enak dibaca).
    ranked.sort()
    return " ".join(sentences[i] for i in ranked).strip()


class ExtractiveMetaReviewGenerator:
    """Generator meta-review ekstraktif (B1): Lead-k atau TextRank.

    Mengimplementasikan Protocol ``MetaReviewGenerator`` (core.interfaces), jadi bisa
    langsung dipasang di pipeline sebagai ganti mock.
    """

    def __init__(
        self,
        method: str = "lead",
        k: int = 3,
        *,
        top_k_aspects: int = 5,
        include_aspect_lead: bool = True,
    ) -> None:
        """``method``: ``"lead"`` atau ``"textrank"``; ``k``: jumlah kalimat ringkasan.

        ``include_aspect_lead``: prepend satu kalimat ringkasan tentang aspek teratas
        (membantu "grounding" di tabel aspek).
        """
        if method not in {"lead", "textrank"}:
            raise ValueError(f"unknown method {method!r}; expected 'lead' or 'textrank'")
        self.method = method
        self.k = k
        self.top_k_aspects = top_k_aspects
        self.include_aspect_lead = include_aspect_lead

    def generate(self, inp: MetaReviewInput) -> str:
        """Ringkas review menjadi satu paragraf meta-review."""
        texts = [r.text for r in inp.reviews if r.text.strip()]
        joined = "\n".join(texts)
        summary = lead_k(joined, self.k) if self.method == "lead" else textrank(joined, self.k)
        if not summary:
            summary = f"No review text available for {inp.game}."
        if self.include_aspect_lead:
            lead = self._aspect_lead(inp)
            if lead:
                summary = f"{lead} {summary}"
        return summary.strip()

    def _aspect_lead(self, inp: MetaReviewInput) -> str:
        """Kalimat pembuka yang menyebut aspek teratas dan polaritasnya."""
        top = [e for e in inp.aspects[: self.top_k_aspects] if e.freq > 0]
        if not top:
            return ""
        names = ", ".join(e.aspect.value for e in top)
        dominant = inp.sentiment.dominant.value
        return (
            f"Reviewers of {inp.game} ({len(inp.reviews)} reviews) focus on {names}, "
            f"with an overall {dominant} reception."
        )
