"""Buat data mock/nyata untuk Task 3 dari hasil Task 1 (Farrel) + Task 2 (Mahesa).

Skema alur:

    reviews.json  --(Task 1 Farrel)--> distribusi sentimen ---\
                                                          +--> MetaReviewInput  --> Task 3 (Lukas)
    reviews.json  --(Task 2 Mahesa)-> tabel aspek -----------/

Script ini memakai **registry pipeline yang sama** dengan CLI (``configs/pipeline.yaml``),
jadi begitu Farrel/Mahesa mengganti ``impl: mock`` dengan implementasi mereka
(``vader``, ``roberta``, ``bertopic``, ...), hasil script ini otomatis jadi output ASLI
mereka, bukan mock lagi.

Contoh:
    # 1) buat payload Task 3 dari review fake
    python experiments/task3_summarization/make_mock_inputs.py \
        --reviews tests/fixtures/sample_reviews.json --game "Fake Game" \
        --out-dir artifacts/runs/mock

    # 2) jalankan model Task 3 terhadap payload itu
    python experiments/task3_summarization/run_inference.py \
        --input artifacts/runs/mock/meta_review_input.json \
        --out artifacts/runs/mock/pipeline_output.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  # pastikan paket repo bisa diimpor tanpa pip install -e

from steam_review_translator.aspect.aggregate import aggregate
from steam_review_translator.aspect.sentences import split_sentences
from steam_review_translator.core.config import load_pipeline_config
from steam_review_translator.core.logging import get_logger
from steam_review_translator.core.schemas import (
    MetaReviewInput,
    PipelineInput,
    reviews_from_json_file,
)
from steam_review_translator.pipeline import build_pipeline

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Argumen CLI."""
    parser = argparse.ArgumentParser(description="Generate Task 3 payload from Task 1 + Task 2")
    parser.add_argument("--reviews", type=Path, required=True, help="JSON list of reviews")
    parser.add_argument("--game", required=True, help="nama game")
    parser.add_argument("--config", type=Path, default=None, help="configs/pipeline.yaml")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/runs/mock"))
    return parser.parse_args()


def main() -> int:
    """Jalankan Task 1 + Task 2, lalu tulis payload Task 3 dan output pipeline."""
    args = parse_args()
    config = load_pipeline_config(args.config)
    reviews = reviews_from_json_file(args.reviews)

    # Pipeline memakai impl yang dipilih di configs/pipeline.yaml (mock / vader / bartopic).
    pipeline = build_pipeline(config)

    # --- Task 1 (Farrel): distribusi sentimen -------------------------------------------
    sentiment = pipeline.sentiment.analyze(reviews)

    # --- Task 2 (Mahesa): tabel aspek ---------------------------------------------------
    sentences = [s for review in reviews for s in split_sentences(review.text)]
    labels = pipeline.tagger.tag(sentences)
    if len(labels) != len(sentences):
        raise ValueError("tagger mengembalikan jumlah label yang tidak sesuai")
    aspects = aggregate(labels)

    # --- Payload Task 3 (Lukas) ---------------------------------------------------------
    meta_input = MetaReviewInput(
        game=args.game, reviews=reviews, sentiment=sentiment, aspects=aspects
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = meta_input.to_json_file(args.out_dir / "meta_review_input.json")
    log.info("payload Task 3 ditulis ke %s", input_path)

    # --- Output lengkap pipeline (kalau Task 3 ikut jalan) -------------------------------
    output = pipeline.run(PipelineInput(game=args.game, reviews=reviews))
    output_path = output.to_json_file(args.out_dir / "pipeline_output.json")
    log.info("output pipeline ditulis ke %s", output_path)

    print(json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
