"""Evaluasi Task 3 di test set (owner: Lukas).

Menghitung (spec bagian 5):
  - ROUGE-1 / ROUGE-2 / ROUGE-L (utama) + ROUGE-Lsum.
  - Cakupan aspek (%).
  - Konsistensi polaritas (opsional).
  - Uji signifikansi paired bootstrap (E1 vs E2, E2 vs E3).

Nilai epoch dijalankan SEKALI di test set (spec: "semua hasil di game test, dijalankan sekali").

Contoh:
    python experiments/task3_summarization/evaluate.py \
        --model artifacts/models/e2_bart --name E2 --aspect-prefix \
        --baseline artifacts/runs/e1_preds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  # pastikan paket repo bisa diimpor tanpa pip install -e
from common import (
    DEFAULT_DATASET,
    build_examples,
    generate_outputs,
    load_split,
    load_task3_config,
    normalize_rows,
    resolve_fields,
)

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.logging import get_logger
from steam_review_translator.core.schemas import AspectEntry
from steam_review_translator.summarization.metrics import (
    aspect_coverage,
    average_rouge,
    paired_bootstrap,
    polarity_consistency,
    rouge_scores,
)

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Argumen CLI evaluasi."""
    parser = argparse.ArgumentParser(description="Evaluate Task 3 meta-review models")
    parser.add_argument("--model", required=True, help="model HF atau path checkpoint")
    parser.add_argument("--name", default="system", help="nama sistem untuk laporan")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="configs/task3/summarization.yaml")
    parser.add_argument("--aspect-prefix", action="store_true", help="pakai prefix aspek (E2/E3)")
    parser.add_argument("--baseline", type=Path, default=None, help="JSON prediksi sistem lain")
    parser.add_argument("--out", type=Path, default=None, help="tulis hasil ke JSON")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--budget-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah sampel")
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--bootstrap", type=int, default=None, help="jumlah resample bootstrap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extractive",
        choices=["lead", "textrank"],
        default=None,
        help="B1: ringkasan ekstraktif (k=3) dari teks input; --model hanya untuk tokenizer",
    )
    parser.add_argument(
        "--records", type=Path, default=None, help="JSON per sampel untuk error_analysis.py"
    )
    return parser.parse_args()


def main() -> int:
    """Jalankan evaluasi dan cetak ringkasan."""
    args = parse_args()
    cfg = load_task3_config(args.config)
    pre_cfg = cfg.get("preprocessing", {})
    asp_cfg = cfg.get("aspects", {})
    dec_cfg = cfg.get("decoding", {})
    inp_cfg = cfg.get("input", {})
    eval_cfg = cfg.get("evaluation", {})

    max_input = args.max_input_tokens or inp_cfg.get("max_input_tokens", 1024)
    max_target = args.max_target_tokens or inp_cfg.get("max_new_tokens", 256)
    budget = args.budget_tokens or pre_cfg.get("budget_tokens", 800)
    top_k = asp_cfg.get("top_k", 5)
    num_beams = args.num_beams or dec_cfg.get("num_beams", 4)
    length_penalty = args.length_penalty or dec_cfg.get("length_penalty", 1.0)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    matcher = LexiconMatcher(load_lexicon())

    rows = load_split(args.split, dataset_name=args.dataset)
    para_key, sum_key, title_key = resolve_fields(list(rows[0].keys()))
    rows = normalize_rows(rows, para_key, sum_key, title_key)
    if args.limit:
        rows = rows[: args.limit]

    examples, stats = build_examples(
        rows,
        tokenizer=tokenizer,
        matcher=matcher,
        budget_tokens=budget,
        max_target_tokens=max_target,
        top_k=top_k,
        use_aspect_prefix=args.aspect_prefix,
    )
    log.info("sampel valid: %d dari %d", stats["valid"], stats["total"])

    if args.extractive:
        from steam_review_translator.summarization.extractive import lead_k, textrank

        pick = lead_k if args.extractive == "lead" else textrank
        k = cfg.get("extractive", {}).get("k", 3)
        predictions = [pick(e.input_text.split("</s>")[-1].strip(), k=k) for e in examples]
    else:
        predictions = generate_outputs(
            args.model,
            examples,
            tokenizer,
            max_input,
            max_target,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
    references = [e.target for e in examples]

    rouge = average_rouge(predictions, references)
    # Bangun ulang tabel aspek tiap sampel (sudah disimpan sebagai dict saat build).
    tables = [[AspectEntry.model_validate(a) for a in e.aspects] for e in examples]
    coverages = [
        aspect_coverage(table, pred, matcher, top_k=top_k)
        for table, pred in zip(tables, predictions, strict=True)
    ]
    consistency = [
        polarity_consistency(table, pred, matcher, top_k=top_k)
        for table, pred in zip(tables, predictions, strict=True)
    ]
    per_sample_rouge = [
        rouge_scores(pred, ref)["rougeL"] for pred, ref in zip(predictions, references, strict=True)
    ]

    report = {
        "system": args.name,
        "model": args.model,
        "n_samples": len(examples),
        "valid_ratio": stats["valid_ratio"],
        "rouge": rouge,
        "aspect_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
        "polarity_consistency": sum(consistency) / len(consistency) if consistency else 0.0,
    }

    # Uji signifikansi paired bootstrap terhadap sistem baseline (E1 vs E2, E2 vs E3).
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        base_scores = baseline.get("per_sample_rougeL") or []
        n = min(len(base_scores), len(per_sample_rouge))
        if n:
            report["bootstrap_vs_baseline"] = paired_bootstrap(
                per_sample_rouge[:n],
                base_scores[:n],
                n_resamples=args.bootstrap or eval_cfg.get("bootstrap_resamples", 1000),
                seed=args.seed,
            )

    print(json.dumps(report, indent=2))
    if args.records:
        args.records.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "game": e.game,
                "input": e.input_text,
                "reference": ref,
                "prediction": pred,
                "aspects": [a["aspect"] for a in e.aspects],
                "rougeL": score,
                "boilerplate_ratio": e.meta["dropped_boilerplate"]
                / max(e.meta["paragraphs_total"], 1),
            }
            for e, pred, ref, score in zip(
                examples, predictions, references, per_sample_rouge, strict=True
            )
        ]
        args.records.write_text(json.dumps(records, indent=1), encoding="utf-8")
        log.info("record per sampel ditulis ke %s", args.records)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({**report, "per_sample_rougeL": per_sample_rouge}, indent=2),
            encoding="utf-8",
        )
        log.info("hasil ditulis ke %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
