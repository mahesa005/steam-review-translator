"""Grow the seed lexicon with terms induced from the train labels (system A).

A unigram or bigram becomes a trigger for an aspect when, over train sentences containing it,
at least ``min_count`` carry that aspect and the share that do is at least ``min_precision``.
Stop words and the seed's ambiguous words are never added. Val/test are not looked at.

Usage (from the repo root):
    python experiments/task2_aspect/systems/build_lexicon.py [min_precision] [min_count]
Writes configs/task2/lexicon_induced.yaml (seed terms + induced terms, with counts as comments).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from steam_review_translator.aspect.lexicon import load_lexicon, tokenize
from steam_review_translator.core.config import project_root
from steam_review_translator.core.schemas import Aspect

SEED = "configs/task2/lexicon.yaml"
OUT = "configs/task2/lexicon_induced.yaml"


def ngrams(tokens: list[str]) -> set[str]:
    grams = {t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2}
    grams |= {
        f"{a} {b}"
        for a, b in zip(tokens, tokens[1:], strict=False)
        if not (a in ENGLISH_STOP_WORDS and b in ENGLISH_STOP_WORDS)
    }
    return grams


def main() -> int:
    min_precision = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    min_count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = load_lexicon(SEED)
    path = project_root() / "data/processed/task2_labels.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    train = [r for r in rows if r["split"] == "train"]

    df: Counter = Counter()
    hits: dict[str, Counter] = defaultdict(Counter)
    for r in train:
        aspects = {x["aspect"] for x in r["labels"]}
        for g in ngrams(tokenize(r["sentence"])):
            df[g] += 1
            for a in aspects:
                hits[g][a] += 1

    induced: dict[Aspect, list[tuple[str, int, float]]] = defaultdict(list)
    for g, n in df.items():
        if g in seed.ambiguous:
            continue
        for a, k in hits[g].items():
            if a == Aspect.OTHER.value or k < min_count:
                continue
            if k / n >= min_precision:
                induced[Aspect(a)].append((g, k, k / n))

    lines = [
        "# Seed lexicon + terms induced from TRAIN labels by",
        "# experiments/task2_aspect/systems/build_lexicon.py",
        f"# (min_precision {min_precision}, min_count {min_count}). Do not edit by hand.",
        "aspects:",
    ]
    total = 0
    for aspect in Aspect:
        seed_terms = list(seed.terms.get(aspect, ()))
        new = sorted(
            (x for x in induced[aspect] if x[0] not in seed_terms), key=lambda x: (-x[1], x[0])
        )
        total += len(new)
        lines.append(f"  {aspect.value}:")
        lines += [f'    - "{t}"' for t in seed_terms]
        lines += [f'    - "{t}"   # {k} sent, precision {p:.2f}' for t, k, p in new]
        if not seed_terms and not new:
            lines[-1] += " []"
    lines.append("ambiguous:")
    lines += [f'  - "{w}"' for w in sorted(seed.ambiguous)]
    out = project_root() / OUT
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{total} induced terms -> {out}")
    for aspect in Aspect:
        print(f"  {aspect.value:<12} +{len(induced[aspect])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
