"""Dataset statistics and system C error analysis for the paper and slides.

Usage (from the repo root; needs the trained model in artifacts/task2/peft):
    python experiments/task2_aspect/systems/report_tables.py
Writes artifacts/task2/report/task2_report_tables.md and errors_peft_test.csv.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict

from steam_review_translator.aspect.metrics import score
from steam_review_translator.aspect.peft_tagger import PeftAspectTagger
from steam_review_translator.core.config import project_root
from steam_review_translator.core.schemas import Aspect, AspectLabel

SPLITS = ["train", "val", "test"]


def fmt(labels: list[dict] | list[AspectLabel]) -> str:
    pairs = [
        (x["aspect"], x["polarity"]) if isinstance(x, dict) else (x.aspect, x.polarity)
        for x in labels
    ]
    return "; ".join(f"{a}/{p}" for a, p in pairs) or "[]"


def table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def main() -> int:
    root = project_root()
    rows = [
        json.loads(x)
        for x in (root / "data/processed/task2_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    md: list[str] = ["# Task 2 tables for the paper and slides", ""]

    # -- dataset statistics -------------------------------------------------------------
    md += ["## Dataset statistics", ""]
    srows = []
    for sp in SPLITS + ["all"]:
        rs = [r for r in rows if sp == "all" or r["split"] == sp]
        n_lab = [len(r["labels"]) for r in rs]
        words = [len(r["sentence"].split()) for r in rs]
        srows.append([
            sp, len({r["game"] for r in rs}), len({r["review_id"] for r in rs}), len(rs),
            f"{sum(n == 0 for n in n_lab) / len(rs):.1%}", f"{statistics.mean(n_lab):.2f}",
            sum(n_lab), f"{statistics.mean(words):.1f}",
        ])  # fmt: skip
    md += table(
        ["split", "games", "reviews", "sentences", "no aspect", "labels/sent", "labels",
         "words/sent"], srows)  # fmt: skip
    md += ["Games per split:", ""]
    for sp in SPLITS:
        games = sorted({r["game"] for r in rows if r["split"] == sp})
        md.append(f"- {sp}: {', '.join(games)}")
    md.append("")

    md += ["## Label distribution (sentences with the aspect; share of positive/negative/neutral)",
           ""]  # fmt: skip
    arows = []
    for a in Aspect:
        cnt = {sp: 0 for sp in SPLITS}
        pol: Counter = Counter()
        for r in rows:
            for x in r["labels"]:
                if x["aspect"] == a.value:
                    cnt[r["split"]] += 1
                    pol[x["polarity"]] += 1
        tot = sum(cnt.values())
        arows.append([a.value, *[cnt[sp] for sp in SPLITS], tot,
                      *[f"{pol[p] / tot:.0%}" if tot else "-" for p in
                        ["positive", "negative", "neutral"]]])  # fmt: skip
    arows.sort(key=lambda r: -r[4])
    md += table(["aspect", "train", "val", "test", "total", "pos", "neg", "neu"], arows)

    md += ["## Training data examples (train split)", ""]
    rng = random.Random(42)
    train = [r for r in rows if r["split"] == "train"]
    picks = [r for r in train if len(r["labels"]) >= 2][:0]
    picks += rng.sample([r for r in train if len(r["labels"]) >= 2], 3)
    picks += rng.sample([r for r in train if len(r["labels"]) == 1], 3)
    picks += rng.sample([r for r in train if not r["labels"]], 2)
    md += table(
        ["game", "sentence", "labels"],
        [[r["game"], r["sentence"].replace("|", "/"), fmt(r["labels"])] for r in picks],
    )

    # -- error analysis of system C on test --------------------------------------------
    test = [r for r in rows if r["split"] == "test"]
    tagger = PeftAspectTagger()
    pred = tagger.tag([r["sentence"] for r in test])
    gold = [[AspectLabel(**x) for x in r["labels"]] for r in test]

    md += ["## System C error analysis (test)", ""]
    kinds: Counter = Counter()
    missed: Counter = Counter()
    extra: Counter = Counter()
    polconf: Counter = Counter()
    errors = []
    for r, g, p in zip(test, gold, pred, strict=True):
        gd = {x.aspect.value: x.polarity.value for x in g}
        pd = {x.aspect.value: x.polarity.value for x in p}
        sent_kinds = []
        for a in gd.keys() - pd.keys():
            missed[a] += 1
            sent_kinds.append("missed aspect")
        for a in pd.keys() - gd.keys():
            extra[a] += 1
            sent_kinds.append("extra aspect")
        for a in gd.keys() & pd.keys():
            polconf[(gd[a], pd[a])] += 1
            if gd[a] != pd[a]:
                sent_kinds.append("wrong polarity")
        if not g and p:
            sent_kinds.append("labels on a no-aspect sentence")
        if g and not p:
            sent_kinds.append("no labels on an aspect sentence")
        kinds.update(set(sent_kinds))
        if sent_kinds:
            errors.append({"sentence_id": r["sentence_id"], "game": r["game"],
                           "sentence": r["sentence"], "gold": fmt(g), "pred": fmt(p),
                           "error_types": ", ".join(sorted(set(sent_kinds)))})  # fmt: skip
    n_err = len(errors)
    md.append(f"{n_err} of {len(test)} test sentences ({n_err / len(test):.1%}) have at least "
              "one error. Sentences per error type (a sentence can have several):")  # fmt: skip
    md.append("")
    md += table(["error type", "sentences"], [[k, v] for k, v in kinds.most_common()])
    md += ["Aspect errors (count of aspect instances):", ""]
    md += table(["aspect", "missed (FN)", "extra (FP)"],
                [[a.value, missed[a.value], extra[a.value]] for a in Aspect])  # fmt: skip
    md += ["Polarity confusion on aspects found by both (rows gold, columns predicted):", ""]
    pols = ["positive", "negative", "neutral"]
    md += table(["gold \\ pred", *pols], [[g, *[polconf[(g, p)] for p in pols]] for g in pols])

    by_game: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(test):
        by_game[r["game"]].append(i)
    md += ["Per game (test):", ""]
    md += table(["game", "sentences", "pair F1", "aspect F1"], [
        [g, len(ix), f"{score([gold[i] for i in ix], [pred[i] for i in ix])['pair']['f1']:.3f}",
         f"{score([gold[i] for i in ix], [pred[i] for i in ix])['aspect']['f1']:.3f}"]
        for g, ix in sorted(by_game.items())])  # fmt: skip

    md += ["Example errors (random, seed 42):", ""]
    sample = random.Random(42).sample(errors, min(12, n_err))
    md += table(["sentence", "gold", "predicted", "error"],
                [[e["sentence"].replace("|", "/"), e["gold"], e["pred"], e["error_types"]]
                 for e in sample])  # fmt: skip

    out = root / "artifacts/task2/report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "task2_report_tables.md").write_text("\n".join(md), encoding="utf-8")
    with (out / "errors_peft_test.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(errors[0]))
        w.writeheader()
        w.writerows(errors)
    print(f"-> {out / 'task2_report_tables.md'}; {n_err} error rows -> errors_peft_test.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
