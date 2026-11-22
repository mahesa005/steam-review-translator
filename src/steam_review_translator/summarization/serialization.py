"""Tahap [C] Task 3: serialisasi input (owner: Lukas).

Format input yang disepakati (spec [C]):

    aspects: gameplay (23, positive) ; story (11, positive) ; performance (8, negative)
    </s> [paragraf terpilih]

Pada **training** tabel aspek berasal dari tabel pseudo (tahap [B]); pada **inference**
tabel yang sama diganti tabel asli dari Task 2. Format-nya identik, sehingga mismatch
train/inference tertutup.

Prefix aspek ini adalah "additional input" yang diuji di skenario E2 (BART + prefix)
dan E3 (T5 + prefix), untuk menjawab RQ2: apakah tabel aspek memperbaiki kualitas ringkasan.
"""

from __future__ import annotations

from collections.abc import Sequence

from steam_review_translator.core.schemas import AspectEntry, MetaReviewInput

#: Pemisah antara prefix aspek dan teks (spec [C]).
SEP_TOKEN = "</s>"

#: Awal prefix aspek (spec [C]).
ASPECTS_PREFIX = "aspects:"


def format_aspect_prefix(aspects: Sequence[AspectEntry], *, top_k: int = 5) -> str:
    """Format tabel aspek menjadi prefix teks, mis. ``aspects: gameplay (23, positive)``.

    Hanya ``top_k`` aspek pertama yang dipakai (urutan input sudah ``freq`` desc).
    Aspek dengan ``freq == 0`` dilewati. Kalau tabel kosong, hasilnya ``"aspects: none"``
    supaya format input tetap konsisten.
    """
    parts = [
        f"{entry.aspect.value} ({entry.freq}, {entry.polarity.value})"
        for entry in aspects[:top_k]
        if entry.freq > 0
    ]
    if not parts:
        return f"{ASPECTS_PREFIX} none"
    return f"{ASPECTS_PREFIX} " + " ; ".join(parts)


def serialize_input(aspect_prefix: str, body: str, *, sep_token: str = SEP_TOKEN) -> str:
    """Gabungkan prefix aspek dan teks terpilih menjadi satu string input.

    Contoh keluaran:
    ``"aspects: gameplay (23, positive) </s> the combat is fun ..."``.
    """
    body = body.strip()
    if not body:
        return aspect_prefix
    return f"{aspect_prefix} {sep_token} {body}"


def build_inference_input(
    inp: MetaReviewInput,
    body: str,
    *,
    top_k: int = 5,
    sep_token: str = SEP_TOKEN,
) -> str:
    """Bangun string input untuk inference dari :class:`MetaReviewInput` Task 3.

    Prefix diambil dari ``inp.aspects`` (tabel asli Task 2), body dari paragraf terpilih.
    """
    prefix = format_aspect_prefix(inp.aspects, top_k=top_k)
    return serialize_input(prefix, body, sep_token=sep_token)
