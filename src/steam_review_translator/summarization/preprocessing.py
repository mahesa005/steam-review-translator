"""Tahap [A] praproses Task 3: pembersihan & seleksi paragraf (owner: Lukas).

Modul ini mengimplementasikan langkah [A] dari rencana eksperimen Task 3:

1. Ambil paragraf dari ``preprocessed_paragraphs_per_reviews`` (sudah lowercase dan
   ter-tokenisasi oleh GameWikiSum, supaya konsisten antara train dan inference).
2. Buang paragraf boilerplate (hasil scraping: header HTML, "enable JavaScript",
   navigasi, footer copyright), paragraf terlalu pendek, dan duplikat antar-dokumen.
3. Skor relevansi tiap paragraf = gabungan kemiripan TF-IDF dengan ``game_title``
   dan jumlah kata kunci lexicon aspek Task 2.
4. Ambil paragraf berskor tertinggi sampai muat ~=800 token (BART) / 400 token (T5),
   sambil mempertahankan urutan dokumen.
5. Tandai sampel yang tersisa < 100 kata sebagai tidak valid (dan laporkan
   persentasenya).

Semua fungsi di modul ini murni Python (tanpa torch / transformers / sklearn) supaya
bisa dipakai dan diuji tanpa extra berat. Tokenizer HF bisa disuntikkan lewat
param ``token_counter`` bila hitungan token yang akurat dibutuhkan.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from steam_review_translator.aspect.lexicon import LexiconMatcher, tokenize

#: Kata kunci tunggal yang menandai paragraf hasil scraping, bukan review (spec [A] langkah 2).
BOILERPLATE_KEYWORDS: frozenset[str] = frozenset(
    {
        "javascript",
        "cookies",
        "cookie",
        "copyright",
        "giveaway",
        "newsletter",
        "advertisement",
        "browser",
        "unsubscribe",
        "webmaster",
        "©",
    }
)

#: Nama perangkat yang tidak mungkin muncul di review game -> satu kemunculan saja
#: sudah cukup untuk menandai paragraf itu berita teknologi, bukan review.
DEVICE_KEYWORDS: frozenset[str] = frozenset(
    {
        "iphone",
        "ipad",
        "macbook",
        "macintosh",
        "smartphone",
        "samsung",
        "airpods",
        "walmart",
        "chromecast",
        "nvidia",
    }
)

#: Kata kunci berita teknologi yang lebih umum (butuh >= 2 kemunculan baru dibuang).
NEWS_TECH_KEYWORDS: frozenset[str] = frozenset(
    {"apple", "android", "screen", "battery", "charger", "pixel", "laptop"}
)

#: Frasa boilerplate (dicari di teks lower-case, sebagai substring).
BOILERPLATE_PHRASES: tuple[str, ...] = (
    "all rights reserved",
    "privacy policy",
    "terms of service",
    "sign up",
    "subscribe to",
    "read more",
    "click here",
    "enable javascript",
    "your browser",
    # Navigasi / berita promo yang ikut ter-scrape (kasus "Spyro" di spec [A]).
    "on sale for",
    "list price",
    "deal of the day",
    "deals of the day",
    "add to cart",
    "free shipping",
    "in stock",
)

#: Paragraf dengan kurang dari 8 kata dibuang (spec [A] langkah 2).
MIN_PARAGRAPH_WORDS = 8

#: Sampel dengan sisa teks < 100 kata setelah pembersihan dibuang (spec [A] langkah 5).
MIN_DOCUMENT_WORDS = 100

#: Bobot TF-IDF vs jumlah keyword aspek dalam skor relevansi.
DEFAULT_ALPHA = 0.6

_WS_RE = re.compile(r"\s+")


def count_words(text: str) -> int:
    """Hitung jumlah kata whitespace-separated pada ``text``."""
    return len(text.split())


def normalize_text(text: str) -> str:
    """Rapatkan spasi berlebih dan potong whitespace di tepi."""
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class Paragraph:
    """Satu paragraf review beserta skor relevansinya."""

    text: str
    score: float = 0.0
    index: int = 0


@dataclass
class PreprocessedDocument:
    """Hasil tahap [A] untuk satu dokumen (satu game).

    ``paragraphs`` adalah paragraf terpilih yang sudah terurut seperti aslinya.
    Statistik jumlah paragraf yang dibuang dipakai untuk laporan coverage (cakupan).
    """

    game: str
    paragraphs: list[str] = field(default_factory=list)
    total_paragraphs: int = 0
    dropped_boilerplate: int = 0
    dropped_short: int = 0
    dropped_duplicate: int = 0
    words: int = 0
    tokens: int = 0

    @property
    def is_valid(self) -> bool:
        """True bila masih tersisa >= 100 kata setelah pembersihan."""
        return self.words >= MIN_DOCUMENT_WORDS

    @property
    def text(self) -> str:
        """Gabungan paragraf terpilih dengan baris baru sebagai pemisah."""
        return "\n".join(self.paragraphs)


class BoilerplateFilter:
    """Buang paragraf boilerplate, terlalu pendek, dan duplikat antar-dokumen."""

    def __init__(
        self,
        keywords: frozenset[str] = BOILERPLATE_KEYWORDS,
        min_words: int = MIN_PARAGRAPH_WORDS,
        phrases: tuple[str, ...] = BOILERPLATE_PHRASES,
        news_keywords: frozenset[str] = NEWS_TECH_KEYWORDS,
        device_keywords: frozenset[str] = DEVICE_KEYWORDS,
    ) -> None:
        """``keywords``: kata pemicu boilerplate; ``min_words``: panjang minimum.

        ``phrases``: frasa boilerplate (dicari sebagai substring pada teks lower-case).
        ``news_keywords``: penanda berita teknologi umum (butuh >= 2 kemunculan).
        ``device_keywords``: nama perangkat (satu kemunculan sudah cukup).
        """
        self.keywords = keywords
        self.min_words = min_words
        self.phrases = phrases
        self.news_keywords = news_keywords
        self.device_keywords = device_keywords

    def is_boilerplate(self, text: str) -> bool:
        """True bila paragraf mengandung kata kunci atau frasa boilerplate."""
        lowered = text.lower()
        if any(phrase in lowered for phrase in self.phrases):
            return True
        tokens = set(tokenize(text))
        if tokens & self.keywords:
            return True
        # Nama perangkat -> pasti berita teknologi, bukan review game.
        if tokens & self.device_keywords:
            return True
        # Kata berita umum: butuh >= 2 kemunculan supaya tidak/doeng.
        return len(tokens & self.news_keywords) >= 2

    def is_too_short(self, text: str) -> bool:
        """True bila paragraf punya kurang dari ``min_words`` kata."""
        return count_words(text) < self.min_words

    def clean(self, paragraphs: Sequence[str]) -> tuple[list[str], dict[str, int]]:
        """Bersihkan daftar paragraf; kembalikan (paragraf tersisa, statistik buang)."""
        stats = {"boilerplate": 0, "short": 0, "duplicate": 0}
        seen: set[str] = set()
        kept: list[str] = []
        for raw in paragraphs:
            text = normalize_text(raw)
            if not text:
                continue
            # Buang duplikat antar-dokumen (duplikat exact, case-insensitive).
            key = " ".join(tokenize(text))
            if key in seen:
                stats["duplicate"] += 1
                continue
            if self.is_boilerplate(text):
                stats["boilerplate"] += 1
                continue
            if self.is_too_short(text):
                stats["short"] += 1
                continue
            seen.add(key)
            kept.append(text)
        return kept, stats


class TfidfIndex:
    """Indeks TF-IDF minimal (pure Python) untuk kemiripan paragraf vs judul game.

    Cukup untuk skor relevansi; tidak butuh scikit-learn supaya modul tetap ringan.
    """

    def __init__(self, documents: Sequence[Sequence[str]]) -> None:
        """Bangun indeks dari token-token tiap dokumen (sudah lowercase)."""
        self._docs = [Counter(doc) for doc in documents]
        self._n = max(len(self._docs), 1)
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        self._idf = {
            term: math.log((self._n + 1) / (count + 1)) + 1.0 for term, count in df.items()
        }
        self._vectors = [self._vectorize(doc) for doc in self._docs]

    def _vectorize(self, doc: Counter[str]) -> dict[str, float]:
        """Ubah counter token menjadi vektor tf-idf ternormalisasi L2."""
        vec = {
            term: (1.0 + math.log(count)) * self._idf.get(term, 1.0) for term, count in doc.items()
        }
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0.0:
            return {}
        return {term: value / norm for term, value in vec.items()}

    def similarities(self, query_tokens: Sequence[str]) -> list[float]:
        """Kosinus kemiripan tiap dokumen terhadap query (judul game)."""
        if not self._docs:
            return []
        query = self._vectorize(Counter(query_tokens))
        if not query:
            return [0.0] * len(self._docs)
        out: list[float] = []
        for vec in self._vectors:
            # Loop di atas dict kecil lebih cepat daripada intersection di sini.
            dot = 0.0
            for term, weight in query.items():
                other = vec.get(term)
                if other is not None:
                    dot += weight * other
            out.append(dot)
        return out


class RelevanceScorer:
    """Skor relevansi paragraf: kemiripan TF-IDF dengan judul + keyword lexicon aspek."""

    def __init__(
        self,
        game_title: str,
        matcher: LexiconMatcher,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        """``alpha``: bobot komponen TF-IDF (0..1); sisanya untuk keyword aspek."""
        self.game_title = game_title
        self.matcher = matcher
        self.alpha = alpha

    def score(self, paragraphs: Sequence[str]) -> list[float]:
        """Kembalikan satu skor relevansi per paragraf (0..1)."""
        if not paragraphs:
            return []
        token_lists = [tokenize(p) for p in paragraphs]
        index = TfidfIndex(token_lists)
        title_tokens = tokenize(self.game_title)
        sims = index.similarities(title_tokens)
        hits = [float(len(self.matcher.match_terms(p))) for p in paragraphs]
        max_hits = max(hits) if hits else 0.0
        scores: list[float] = []
        for sim, hit in zip(sims, hits, strict=True):
            aspect_part = (hit / max_hits) if max_hits > 0 else 0.0
            scores.append(self.alpha * sim + (1.0 - self.alpha) * aspect_part)
        return scores


class ParagraphSelector:
    """Pilih paragraf berskor tertinggi sampai budget token, lalu urutkan seperti asli."""

    def __init__(self, token_counter: Callable[[str], int] | None = None) -> None:
        """``token_counter``: fungsi menghitung token; default = jumlah kata."""
        self.token_counter = token_counter or count_words

    def select(self, paragraphs: Sequence[Paragraph], budget_tokens: int) -> list[str]:
        """Pilih paragraf dalam budget dan kembalikan urutannya seperti dokumen asli."""
        if budget_tokens <= 0 or not paragraphs:
            return []
        # Urut berdasarkan skor (desc), lalu indeks (asc) supaya tie-break deterministik.
        ranked = sorted(paragraphs, key=lambda p: (-p.score, p.index))
        chosen: list[Paragraph] = []
        used = 0
        for para in ranked:
            cost = self.token_counter(para.text)
            if used + cost > budget_tokens:
                continue  # coba paragraf lain yang lebih pendek
            chosen.append(para)
            used += cost
        # Kembalikan ke urutan dokumen (konsistensi naratif).
        chosen.sort(key=lambda p: p.index)
        return [p.text for p in chosen]


def preprocess_document(
    game: str,
    paragraphs: Sequence[str],
    *,
    matcher: LexiconMatcher,
    budget_tokens: int = 800,
    token_counter: Callable[[str], int] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> PreprocessedDocument:
    """Jalankan seluruh tahap [A] untuk satu dokumen.

    Mengembalikan :class:`PreprocessedDocument` berisi paragraf terpilih, jumlah kata,
    jumlah token, dan statistik paragraf yang dibuang.
    """
    cleaner = BoilerplateFilter()
    kept, stats = cleaner.clean(paragraphs)
    scorer = RelevanceScorer(game, matcher, alpha=alpha)
    raw_scores = scorer.score(kept)
    scored = [
        Paragraph(text=text, score=score, index=i)
        for i, (text, score) in enumerate(zip(kept, raw_scores, strict=True))
    ]
    selector = ParagraphSelector(token_counter=token_counter)
    selected = selector.select(scored, budget_tokens)
    words = sum(count_words(p) for p in selected)
    tokens = sum((token_counter or count_words)(p) for p in selected)
    return PreprocessedDocument(
        game=game,
        paragraphs=selected,
        total_paragraphs=len(paragraphs),
        dropped_boilerplate=stats["boilerplate"],
        dropped_short=stats["short"],
        dropped_duplicate=stats["duplicate"],
        words=words,
        tokens=tokens,
    )


def preprocess_reviews(
    game: str,
    reviews: Sequence[str],
    *,
    matcher: LexiconMatcher,
    budget_tokens: int = 800,
    token_counter: Callable[[str], int] | None = None,
) -> PreprocessedDocument:
    """Versi ringkas: gabungkan beberapa review (satu game) jadi satu dokumen."""
    paragraphs: list[str] = []
    for review in reviews:
        paragraphs.extend(part for part in review.split("\n") if part.strip())
    return preprocess_document(
        game,
        paragraphs,
        matcher=matcher,
        budget_tokens=budget_tokens,
        token_counter=token_counter,
    )
