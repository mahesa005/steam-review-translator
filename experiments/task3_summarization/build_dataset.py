"""Ubah data GameWikiSum (data/raw/Full_GameWiki) menjadi dataset Task 3.

File sumber: train.json / val.json / test.json (2,7 GB total). Strukturnya:
    [ [ {game_id, game_title, platform, wiki_title, summary,
         preprocessed_summary, preprocessed_game_title,
         preprocessed_paragraphs_per_reviews, raw_reviews}, ... ], ... ]
yaitu array luar = kelompok (batch), array dalam = dokumen. Satu DOKUMEN = satu sampel
(1.454 dokumen untuk split test, sesuai split resmi GameWikiSum).

Yang dilakukan script ini (mengikuti spec Task 3):
  [A] bersihkan boilerplate + pilih paragraf relevan sampai budget token
  [B] tabel aspek pseudo  -> lexicon Task 2 (frekuensi) + sentimen Task 1 (polaritas)
  [C] serialisasi input   -> "aspects: gameplay (23, positive) ; ... </s> <teks>"
  target                -> preprocessed_summary, dipotong ke 256 token

Tabel aspek di training sengaja memakai PSEUDO table (spec [B]): pada inference
tabel yang sama diganti tabel ASLI Task 2, dengan format identik.

Contoh:
    python experiments/task3_summarization/build_dataset.py \
        --split train --out data/interim/task3/train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  # pastikan paket repo bisa diimpor tanpa pip install -e
from common import iter_json_array

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.logging import get_logger
from steam_review_translator.summarization.aspects import make_polarity_fn, pseudo_aspect_table
from steam_review_translator.summarization.preprocessing import preprocess_document
from steam_review_translator.summarization.serialization import (
    format_aspect_prefix,
    serialize_input,
)

log = get_logger(__name__)

#: Lokasi dataset GameWikiSum lokal (tidak di-commit ke git).
DEFAULT_DATA_DIR = "data/raw/Full_GameWiki"

#: Nama file per split.
SPLIT_FILES = {"train": "train.json", "val": "val.json", "test": "test.json"}


def parse_args() -> argparse.Namespace:
    """Argumen CLI."""
    parser = argparse.ArgumentParser(description="Build the Task 3 dataset from GameWikiSum")
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="train")
    parser.add_argument("--out", type=Path, required=True, help="tujuan JSONL")
    parser.add_argument(
        "--budget-tokens", type=int, default=800, help="budget [A] (800 BART / 400 T5)"
    )
    parser.add_argument("--max-target-tokens", type=int, default=256, help="panjang target")
    parser.add_argument("--top-k", type=int, default=5, help="jumlah aspek pada prefix")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah game (uji cepat)")
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=2000,
        help="batas paragraf per game (game besar punya >60k paragraf)",
    )
    parser.add_argument("--no-aspect-prefix", action="store_true", help="E1: tanpa prefix aspek")
    parser.add_argument("--lexicon", type=Path, default=None, help="path lexicon Task 2")
    parser.add_argument("--sentiment-impl", default="mock", help="impl Task 1 (mock/vader/...)")
    return parser.parse_args()


def build_sentiment_polarity_fn(impl: str) -> Any:
    """Buat fungsi polaritas kalimat dari implementasi Task 1 yang dipilih.

    Default ``mock`` (keyword + flag ``recommended``). Begitu Farrel memasang model
    aslinya, ganti dengan ``vader`` dsb. (lihat ``configs/pipeline.yaml``).
    """
    if impl == "mock":
        from steam_review_translator.sentiment.mock import MockSentimentAnalyzer

        analyzer = MockSentimentAnalyzer()
    else:
        from steam_review_translator.pipeline import SENTIMENT_REGISTRY, _import

        if impl not in SENTIMENT_REGISTRY:
            raise ValueError(f"impl Task 1 {impl!r} tidak terdaftar: {sorted(SENTIMENT_REGISTRY)}")
        analyzer = _import(SENTIMENT_REGISTRY[impl])()
    return make_polarity_fn(analyzer)


def paragraphs_of_document(doc: dict[str, Any], max_paragraphs: int | None = None) -> list[str]:
    """Kumpulkan semua paragraf dari satu dokumen (semua review di dalamnya).

    ``max_paragraphs`` membatasi jumlah paragraf yang diproses (beberapa dokumen punya
    ribuan paragraf; memproses semuanya terlalu lambat).
    """
    paragraphs: list[str] = []
    for review in doc.get("preprocessed_paragraphs_per_reviews") or []:
        if isinstance(review, str):
            paragraphs.append(review)
        else:
            paragraphs.extend(review)
        if max_paragraphs and len(paragraphs) >= max_paragraphs:
            return paragraphs[:max_paragraphs]
    return paragraphs


def target_of_document(doc: dict[str, Any]) -> str:
    """Ambil ringkasan referensi satu dokumen (prefer ``preprocessed_summary``)."""
    return doc.get("preprocessed_summary") or doc.get("summary") or ""


def title_of_document(doc: dict[str, Any]) -> str:
    """Judul game satu dokumen (fallback ke judul wiki)."""
    return str(doc.get("game_title") or doc.get("wiki_title") or "unknown game")


def truncate_words(text: str, max_tokens: int) -> str:
    """Potong target ke ~max_tokens token.

    Dipakai kalau tokenizer HF tidak tersedia. Rasio 0,75 kata/token cukup dekat
    untuk tokenizer BART/T5 pada teks Inggris.
    """
    words = text.split()
    limit = max(int(max_tokens * 0.75), 1)
    return " ".join(words[:limit])


def main() -> int:
    """Bangun dataset Task 3 dari satu split GameWikiSum."""
    args = parse_args()
    source = args.data_dir / SPLIT_FILES[args.split]
    if not source.is_file():
        raise FileNotFoundError(f"{source} tidak ditemukan")

    matcher = LexiconMatcher(load_lexicon(args.lexicon))
    polarity_fn = build_sentiment_polarity_fn(args.sentiment_impl)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    dropped = 0
    with args.out.open("w", encoding="utf-8") as fh:
        # Struktur file: array luar = kelompok (batch), array dalam = dokumen.
        # Satu DOKUMEN = satu sampel (1.454 dokumen untuk split test, sesuai split resmi).
        for group in iter_json_array(source):
            if not isinstance(group, list):
                continue
            for document in group:
                if not isinstance(document, dict):
                    continue
                total += 1
                game_title = title_of_document(document)
                paragraphs = paragraphs_of_document(document, args.max_paragraphs)
                target = target_of_document(document)

                # [A] pembersihan + seleksi paragraf
                doc = preprocess_document(
                    game_title,
                    paragraphs,
                    matcher=matcher,
                    budget_tokens=args.budget_tokens,
                )
                if not doc.is_valid:
                    dropped += 1  # spec [A] langkah 5: < 100 kata -> dibuang
                    continue

                # [B] tabel aspek pseudo (training) -> prefix
                table = pseudo_aspect_table(doc.text, matcher, polarity_fn, top_k=args.top_k)
                prefix = format_aspect_prefix(table, top_k=args.top_k)

                # [C] serialisasi input
                input_text = (
                    doc.text if args.no_aspect_prefix else serialize_input(prefix, doc.text)
                )

                row = {
                    "game": game_title,
                    "input": input_text,
                    "target": truncate_words(target, args.max_target_tokens),
                    "aspects": [e.model_dump(mode="json") for e in table],
                    "meta": {
                        "words": doc.words,
                        "paragraphs_total": doc.total_paragraphs,
                        "paragraphs_kept": len(doc.paragraphs),
                        "dropped_boilerplate": doc.dropped_boilerplate,
                        "dropped_short": doc.dropped_short,
                        "dropped_duplicate": doc.dropped_duplicate,
                        "game_id": document.get("game_id"),
                        "platform": document.get("platform"),
                        "wiki_title": document.get("wiki_title"),
                        "n_reviews": len(document.get("preprocessed_paragraphs_per_reviews") or []),
                    },
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                if args.limit and kept >= args.limit:
                    break
            if args.limit and kept >= args.limit:
                break

    ratio = 100 * kept / total if total else 0.0
    log.info(
        "split %s: %d dokumen -> %d sampel valid (%.1f%%), %d dibuang (<100 kata) -> %s",
        args.split,
        total,
        kept,
        ratio,
        dropped,
        args.out,
    )
    print(f"{args.out}: {kept}/{total} dokumen dipakai ({ratio:.1f}%), {dropped} dibuang")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
