"""Metrik evaluasi Task 3 (owner: Lukas).

Sesuai spec bagian 5 Evaluasi:

- **ROUGE-1, ROUGE-2, ROUGE-L** (metrik utama) + ROUGE-Lsum.
- **Cakupan aspek** : persentase aspek top-k pada tabel yang disebut di keluaran
  (pencocokan lexicon). Metrik ini mengukur apakah prefix aspek benar-benar dipakai model.
- **Konsistensi polaritas** : kesesuaian polaritas aspek yang disebut di keluaran
  dengan polaritas di tabel (opsional di spec).

Implementasi memakai ``rouge_score`` bila terpasang (extra ``.[task3]``); kalau tidak,
ada fallback pure Python (F1 n-gram + LCS) supaya modul tetap bisa dipakai di CI
yang hanya memasang ``.[dev]``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from steam_review_translator.aspect.lexicon import LexiconMatcher
from steam_review_translator.aspect.sentences import split_sentences
from steam_review_translator.core.schemas import AspectEntry, Polarity

#: Kata sentimen minimal untuk mengukur konsistensi polaritas.
POSITIVE_WORDS = frozenset(
    {
        "good",
        "great",
        "amazing",
        "awesome",
        "love",
        "loved",
        "loves",
        "lovely",
        "fun",
        "excellent",
        "fantastic",
        "gorgeous",
        "solid",
        "enjoy",
        "enjoyed",
        "best",
        "perfect",
        "masterpiece",
        "recommend",
        "incredible",
        "clever",
        "fast",
        "wonderful",
        "brilliant",
        "superb",
        "refreshing",
        "addictive",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "terrible",
        "awful",
        "boring",
        "hate",
        "hated",
        "broken",
        "worst",
        "poor",
        "crash",
        "crashes",
        "crashing",
        "lag",
        "laggy",
        "stutter",
        "expensive",
        "disappointing",
        "annoying",
        "overpriced",
        "refund",
        "repetitive",
        "grindy",
        "unplayable",
        "mediocre",
        "bland",
        "buggy",
        "bug",
        "bugs",
        "waste",
    }
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    """Token kata lowercase."""
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Buat daftar n-gram dari token."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _f1(match: int, total_pred: int, total_ref: int) -> float:
    """Hitung F1 dari jumlah match, jumlah prediksi, jumlah referensi."""
    if match == 0 or total_pred == 0 or total_ref == 0:
        return 0.0
    precision = match / total_pred
    recall = match / total_ref
    return 2 * precision * recall / (precision + recall)


def _rouge_n(pred: str, ref: str, n: int) -> float:
    """ROUGE-N (F1) versi pure Python."""
    pred_grams = _ngrams(_tokens(pred), n)
    ref_grams = _ngrams(_tokens(ref), n)
    if not pred_grams or not ref_grams:
        return 0.0
    ref_pool: dict[tuple[str, ...], int] = {}
    for gram in ref_grams:
        ref_pool[gram] = ref_pool.get(gram, 0) + 1
    match = 0
    for gram in pred_grams:
        if ref_pool.get(gram, 0) > 0:
            ref_pool[gram] -= 1
            match += 1
    return _f1(match, len(pred_grams), len(ref_grams))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Panjang longest common subsequence (untuk ROUGE-L)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                curr.append(prev[j] + 1)
            else:
                curr.append(max(curr[j], prev[j + 1]))
        prev = curr
    return prev[-1]


def _rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L (F1) versi pure Python berbasis LCS."""
    a, b = _tokens(pred), _tokens(ref)
    lcs = _lcs_length(a, b)
    return _f1(lcs, len(a), len(b))


def _rouge_lsum(pred: str, ref: str) -> float:
    """ROUGE-Lsum: ROUGE-L per kalimat, dirata-rata (union-LCS antar kalimat).

    Versi sederhana: hitung LCS gabungan dengan memisahkan kalimat memakai newline.
    """
    a = [t for s in pred.split("\n") for t in _tokens(s)]
    b = [t for s in ref.split("\n") for t in _tokens(s)]
    lcs = _lcs_length(a, b)
    return _f1(lcs, len(a), len(b))


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """Hitung ROUGE-1/2/L/Lsum untuk satu pasangan (prediksi, referensi).

    Memakai ``rouge_score`` bila tersedia, else fallback pure Python.
    """
    try:  # pragma: no cover - bergantung pada environment
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
            "rougeLsum": scores["rougeLsum"].fmeasure,
        }
    except ImportError:
        return {
            "rouge1": _rouge_n(prediction, reference, 1),
            "rouge2": _rouge_n(prediction, reference, 2),
            "rougeL": _rouge_l(prediction, reference),
            "rougeLsum": _rouge_lsum(prediction, reference),
        }


def average_rouge(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Rata-rata ROUGE atas seluruh pasangan (harus panjang sama)."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) must match"
        )
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    for pred, ref in zip(predictions, references, strict=True):
        for key, value in rouge_scores(pred, ref).items():
            totals[key] += value
    return {key: value / len(predictions) for key, value in totals.items()}


def aspect_coverage(
    aspects: Sequence[AspectEntry],
    output: str,
    matcher: LexiconMatcher,
    *,
    top_k: int = 5,
) -> float:
    """Persentase aspek top-k pada tabel yang disebut di dalam keluaran.

    Mengukur apakah prefix aspek benar-benar dipakai model (spec Evaluasi).
    Hasil 0.0 kalau tabel kosong.
    """
    mentioned = set(matcher.match(output))
    targets = [e for e in aspects[:top_k] if e.freq > 0]
    if not targets:
        return 0.0
    hits = sum(1 for e in targets if e.aspect in mentioned)
    return hits / len(targets)


def _sentence_polarity(sentence: str) -> Polarity:
    """Polaritas kasar satu kalimat dari kamus kata sentiment (skala 1-5)."""
    tokens = set(_tokens(sentence))
    score = len(tokens & POSITIVE_WORDS) - len(tokens & NEGATIVE_WORDS)
    if score > 0:
        return Polarity.POSITIVE
    if score < 0:
        return Polarity.NEGATIVE
    return Polarity.NEUTRAL


def polarity_consistency(
    aspects: Sequence[AspectEntry],
    output: str,
    matcher: LexiconMatcher,
    *,
    top_k: int = 5,
) -> float:
    """Tingkat kesesuaian polaritas aspek yang disebut keluaran vs tabel.

    Untuk tiap aspek yang disebut di keluaran, polaritas kalimatnya dibandingkan
    dengan polaritas di tabel. Mengembalikan fraksi yang cocok (0..1); 0.0 kalau tidak
    ada aspek yang bisa dibandingkan.
    """
    table = {e.aspect: e.polarity for e in aspects[:top_k] if e.freq > 0}
    if not table:
        return 0.0
    sentences = split_sentences(output)
    total = 0
    agree = 0
    for sentence in sentences:
        mentioned = [a for a in matcher.match(sentence) if a in table]
        if not mentioned:
            continue
        predicted = _sentence_polarity(sentence)
        for aspect in mentioned:
            total += 1
            # Kalimat netral tidak mengkontradiksi tabel: dianggap konsisten.
            if predicted is Polarity.NEUTRAL or predicted is table[aspect]:
                agree += 1
    return agree / total if total else 0.0


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Uji signifikansi paired bootstrap untuk dua sistem (spec: E1 vs E2, E2 vs E3).

    Mengembalikan mean selisih, p-value dua arah, dan interval keyakinan 95%.
    """
    import random

    if len(scores_a) != len(scores_b):
        raise ValueError("score lists must have the same length")
    if not scores_a:
        raise ValueError("score lists must not be empty")
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    count = 0
    samples: list[float] = []
    for _ in range(n_resamples):
        sample = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        samples.append(sample)
        if abs(sample) >= abs(observed):
            count += 1
    samples.sort()
    lo = samples[int(0.025 * (n_resamples - 1))]
    hi = samples[int(0.975 * (n_resamples - 1))]
    return {
        "mean_diff": observed,
        "p_value": (count + 1) / (n_resamples + 1),
        "ci_low": lo,
        "ci_high": hi,
    }
