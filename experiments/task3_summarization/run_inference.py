"""Jalankan model Task 3 terhadap payload MetaReviewInput, lalu tampilkan outputnya.

Ini skrip "review output": terima file ``meta_review_input.json`` (hasil Task 1 + Task 2),
jalankan generator Task 3 (ekstraktif tanpa ML, atau BART/T5 bila checkpoint tersedia),
lalu tulis ``pipeline_output.json`` sesuai kontrak bersama.

Contoh:
    # tanpa ML (baseline B1 Lead-k / TextRank)
    python experiments/task3_summarization/run_inference.py \
        --input artifacts/runs/mock/meta_review_input.json

    # dengan model BART hasil fine-tuning (E2)
    python experiments/task3_summarization/run_inference.py \
        --input artifacts/runs/mock/meta_review_input.json \
        --model artifacts/models/e2_bart --aspect-prefix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  # pastikan paket repo bisa diimpor tanpa pip install -e

from steam_review_translator.core.logging import get_logger
from steam_review_translator.core.schemas import MetaReviewInput, PipelineOutput
from steam_review_translator.summarization.extractive import ExtractiveMetaReviewGenerator
from steam_review_translator.summarization.transformer import (
    BartMetaReviewGenerator,
    T5MetaReviewGenerator,
)

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Argumen CLI."""
    parser = argparse.ArgumentParser(description="Run Task 3 on a MetaReviewInput JSON")
    parser.add_argument("--input", type=Path, required=True, help="meta_review_input.json")
    parser.add_argument("--out", type=Path, default=None, help="tulis pipeline_output.json")
    parser.add_argument("--model", type=Path, default=None, help="checkpoint BART/T5 (opsional)")
    parser.add_argument("--architecture", choices=["bart", "t5"], default="bart")
    parser.add_argument("--aspect-prefix", action="store_true", help="pakai prefix aspek (E2/E3)")
    parser.add_argument(
        "--method", choices=["lead", "textrank"], default="textrank", help="mode B1"
    )
    parser.add_argument("--k", type=int, default=3, help="jumlah kalimat untuk baseline B1")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k-aspects", type=int, default=5)
    return parser.parse_args()


def build_generator(args: argparse.Namespace) -> object:
    """Pilih generator: model checkpoint (BART/T5) atau baseline ekstraktif (B1)."""
    if args.model is not None:
        cls = BartMetaReviewGenerator if args.architecture == "bart" else T5MetaReviewGenerator
        return cls(
            model_name=str(args.model),
            checkpoint=args.model,
            use_aspect_prefix=args.aspect_prefix,
            top_k_aspects=args.top_k_aspects,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            max_new_tokens=args.max_new_tokens,
        )
    return ExtractiveMetaReviewGenerator(method=args.method, k=args.k)


def main() -> int:
    """Jalankan Task 3 dan tampilkan hasilnya."""
    args = parse_args()
    inp = MetaReviewInput.from_json_file(args.input)
    generator = build_generator(args)
    meta_review = generator.generate(inp)

    output = PipelineOutput(
        game=inp.game,
        sentiment=inp.sentiment,
        aspects=inp.aspects,
        meta_review=meta_review,
    )
    print(json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False))
    if args.out is not None:
        path = output.to_json_file(args.out)
        log.info("output ditulis ke %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
