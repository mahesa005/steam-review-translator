"""Tahap [B] Task 3: tabel aspek pseudo untuk data training (owner: Lukas).

GameWikiSum tidak menyediakan tabel aspek, jadi tabel harus dibangun sendiri dari teks
train. Sesuai spesifikasi:

- **Frekuensi** = jumlah kemunculan kata kunci lexicon aspek Task 2.
- **Polaritas** = model sentimen Task 1 (Farrel) pada kalimat yang memuat aspek.
- Hasil akhir diambil **top-5 aspek** (urut frekuensi menurun).

Modul ini sengaja hanya bergantung pada ``core.schemas`` + paket ``aspect`` (lexicon
Task 2) + ``core.interfaces`` (model sentimen Task 1), sehingga tidak ada dependensi
antara task selain kontrak JSON yang sudah disepakati.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from steam_review_translator.aspect.aggregate import aggregate
from steam_review_translator.aspect.lexicon import LexiconMatcher
from steam_review_translator.aspect.sentences import split_sentences
from steam_review_translator.core.interfaces import SentimentAnalyzer
from steam_review_translator.core.schemas import AspectEntry, AspectLabel, Polarity, Review

#: Jumlah aspek yang dipakai pada prefix input (spec [B] dan [C]).
DEFAULT_TOP_K = 5

#: Fungsi polaritas tingkat kalimat: kalimat -> polaritas.
SentencePolarityFn = Callable[[str], Polarity]


def make_polarity_fn(analyzer: SentimentAnalyzer) -> SentencePolarityFn:
    """Bungkus model sentimen Task 1 menjadi fungsi polaritas per kalimat.

    Model Task 1 bekerja per batch review; di sini tiap kalimat dijalankan satu per
    satu dan label dominannya dipakai. Hasilnya di-cache per kalimat supaya kalimat
    yang sama tidak dihitung dua kali.
    """
    cache: dict[str, Polarity] = {}

    def polarity_fn(sentence: str) -> Polarity:
        # Cache dulu: kalimat sering berulang antar review.
        cached = cache.get(sentence)
        if cached is not None:
            return cached
        result = analyzer.analyze([Review(review_id="snippet", text=sentence)])
        polarity = result.dominant
        cache[sentence] = polarity
        return polarity

    return polarity_fn


def lexicon_polarity_fn(sentence: str) -> Polarity:
    """Fallback sederhana tanpa model: polaritas netral untuk semua kalimat.

    Dipakai ketika model sentimen Task 1 belum tersedia; polaritas tabel pseudo
    akan selalu ``neutral`` (freq tetap valid).
    """
    del sentence
    return Polarity.NEUTRAL


def pseudo_aspect_table(
    text: str,
    matcher: LexiconMatcher,
    polarity_fn: SentencePolarityFn = lexicon_polarity_fn,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[AspectEntry]:
    """Bangun tabel aspek pseudo dari satu teks (satu dokumen train).

    Setiap kalimat dipecah, tiap aspek yang muncul di kalimat itu di-label dengan
    polaritas kalimat tersebut, lalu diagregasi dengan aturan Task 2 yang sama
    (``aspect.aggregate.aggregate``). Hasil dipotong jadi ``top_k`` aspek teratas.
    """
    sentences = split_sentences(text)
    labels: list[list[AspectLabel]] = []
    for sentence in sentences:
        aspects = matcher.match(sentence)
        if not aspects:
            labels.append([])
            continue
        polarity = polarity_fn(sentence)
        labels.append([AspectLabel(aspect=aspect, polarity=polarity) for aspect in aspects])
    table = aggregate(labels)
    return table[:top_k]


def pseudo_aspect_table_from_reviews(
    reviews: Sequence[str],
    matcher: LexiconMatcher,
    polarity_fn: SentencePolarityFn = lexicon_polarity_fn,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[AspectEntry]:
    """Bangun tabel aspek pseudo dari beberapa review (satu game)."""
    joined = "\n".join(reviews)
    return pseudo_aspect_table(joined, matcher, polarity_fn, top_k=top_k)
