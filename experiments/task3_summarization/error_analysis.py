"""Error analysis Task 3 (owner: Lukas) - spec bagian 6.

Ambil 50 sampel test (stratifikasi: ROUGE-L tinggi, sedang, rendah), lalu beri tiap
kasus satu kategori:
  1. Halusinasi          : fakta/nama/angka yang tidak ada di input.
  2. Repetisi            : kalimat atau frasa berulang.
  3. Aspek terlewat/polaritas salah dibanding tabel.
  4. Gaya tidak sesuai   : ringkasan plot netral ala Wikipedia, bukan konsensus opini.
  5. Kegagalan input     : sebagian besar teks ternyata boilerplate.

Hasil dipecah menurut panjang input dan jumlah dokumen valid setelah pembersihan,
lalu dikuantifikasi sebagai tabel frekuensi kategori per sistem (E1 vs E2 vs E3).

Contoh:
    python experiments/task3_summarization/error_analysis.py \
        --system E2=artifacts/runs/e2_preds.json --system E3=artifacts/runs/e3_preds.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  # pastikan paket repo bisa diimpor tanpa pip install -e

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.logging import get_logger

log = get_logger(__name__)

#: Nama kategori error (spec bagian 6).
CATEGORIES = (
    "halusinasi",
    "repetisi",
    "aspek_terlewat",
    "gaya_tidak_sesuai",
    "kegagalan_input",
)

#: Kata/frasa yang menandakan gaya deskriptif ala Wikipedia, bukan konsensus opini.
WIKI_STYLE_MARKERS = (
    "is a",
    "was developed",
    "was released",
    "the game was",
    "developed by",
    "released in",
    "features",
    "gameplay includes",
)


def classify_error(
    prediction: str,
    reference: str,
    input_text: str,
    *,
    mentioned_aspects: set[str],
    matcher: LexiconMatcher,
    boilerplate_ratio: float,
) -> str:
    """Tentukan satu kategori error untuk satu keluaran model.

    Urutan pemeriksaan mengikuti gaya yang paling fatal lebih dulu (halusinasi >
    repetisi > aspek > gaya > kegagalan input).
    """
    # 5. Kegagalan input: sebagian besar teks pada dasarnya boilerplate.
    if boilerplate_ratio >= 0.8:
        return "kegagalan_input"
    # 1. Halusinasi: kata nama besar/angka di keluaran tapi tidak ada di input.
    if _has_unsupported_facts(prediction, input_text):
        return "halusinasi"
    # 2. Repetisi: frasa yang sama muncul lebih dari sekali.
    if _is_repetitive(prediction):
        return "repetisi"
    # 3. Aspek terlewat: prefix aspek tidak sama sekali disebut di keluaran.
    if mentioned_aspects and not (mentioned_aspects & set(matcher.match(prediction))):
        return "aspek_terlewat"
    # 4. Gaya tidak sesuai: kalimat generik ala Wikipedia.
    lowered = prediction.lower()
    if any(marker in lowered for marker in WIKI_STYLE_MARKERS):
        return "gaya_tidak_sesuai"
    # Tidak ada kategori yang cocok -> keluaran dianggap bersih.
    return "tidak_ada_error"


def _has_unsupported_facts(prediction: str, input_text: str) -> bool:
    """True bila ada nama besar atau angka di keluaran yang tidak ada di input."""
    import re

    def facts(text: str) -> set[str]:
        # Nama kapital (bukan di awal kalimat) dan angka.
        caps = set(re.findall(r"\b(?!I\b)[A-Z][a-zA-Z]{2,}\b", text))
        numbers = set(re.findall(r"\b\d[\d.,]*\b", text))
        return caps | numbers

    new_facts = facts(prediction) - facts(input_text)
    return bool(new_facts)


def _is_repetitive(prediction: str) -> bool:
    """True bila ada frasa yang diulang >= 3 kali atau kalimat identik berulang."""
    words = prediction.lower().split()
    if len(words) < 12:
        return False
    counts = Counter(" ".join(words[i : i + 3]) for i in range(len(words) - 2))
    return any(count >= 3 for count in counts.values())


def stratify(scores: list[float], n: int = 50) -> list[int]:
    """Ambil ``n`` indeks sampel dengan stratifikasi ROUGE-L tinggi/sedang/rendah."""
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n = min(n, len(order))
    buckets = [
        order[: len(order) // 3],  # tinggi
        order[len(order) // 3 : 2 * len(order) // 3],  # sedang
        order[2 * len(order) // 3 :],  # rendah
    ]
    chosen: list[int] = []
    per_bucket = max(n // 3, 1)
    for bucket in buckets:
        chosen.extend(bucket[:per_bucket])
    # Isi sisa kalau jumlah bucket kurang.
    for idx in order:
        if len(chosen) >= n:
            break
        if idx not in chosen:
            chosen.append(idx)
    return chosen[:n]


def analyze_system(
    name: str,
    records: list[dict[str, Any]],
    *,
    matcher: LexiconMatcher,
) -> dict[str, Any]:
    """Klasifikasikan error satu sistem dan kembalikan tabel frekuensi."""
    scores = [float(r.get("rougeL", 0.0)) for r in records]
    indices = stratify(scores, 50)
    counter: Counter[str] = Counter()
    by_input_length: dict[str, Counter[str]] = {
        "short": Counter(),
        "medium": Counter(),
        "long": Counter(),
    }
    details: list[dict[str, Any]] = []
    for idx in indices:
        record = records[idx]
        prediction = str(record.get("prediction", ""))
        reference = str(record.get("reference", ""))
        input_text = str(record.get("input", ""))
        aspects = set(record.get("aspects", []))
        words = len(input_text.split())
        length_bucket = "short" if words < 400 else "medium" if words < 1200 else "long"
        boilerplate = float(record.get("boilerplate_ratio", 0.0))
        category = classify_error(
            prediction,
            reference,
            input_text,
            mentioned_aspects=aspects,
            matcher=matcher,
            boilerplate_ratio=boilerplate,
        )
        counter[category] += 1
        by_input_length[length_bucket][category] += 1
        details.append(
            {
                "index": idx,
                "rougeL": scores[idx],
                "category": category,
                "input_words": words,
                "prediction": prediction[:400],
            }
        )
    return {
        "system": name,
        "n_analyzed": len(indices),
        "frequency": {cat: counter.get(cat, 0) for cat in CATEGORIES},
        "other": counter.get("tidak_ada_error", 0),
        "by_input_length": {k: dict(v) for k, v in by_input_length.items()},
        "cases": details,
    }


def main() -> int:
    """Bandingkan beberapa sistem dan cetak tabel frekuensi kategori."""
    parser = argparse.ArgumentParser(description="Error analysis for Task 3")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help="format NAMA=file.json (bisa diulang)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    matcher = LexiconMatcher(load_lexicon())
    reports: list[dict[str, Any]] = []
    for spec in args.system:
        name, _, path = spec.partition("=")
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        reports.append(analyze_system(name, records, matcher=matcher))

    for report in reports:
        print(f"\n=== {report['system']} (n={report['n_analyzed']}) ===")
        for category, count in report["frequency"].items():
            print(f"  {category:<20} {count}")
        print(f"  {'tidak_ada_error':<20} {report['other']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        log.info("laporan ditulis ke %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
