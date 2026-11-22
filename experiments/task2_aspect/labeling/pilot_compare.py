"""Pilot: Haiku vs Sonnet as reviewers of Opus labels, on 20 train reviews (Claude Code agents).

Usage (from the repo root):
    python experiments/task2_aspect/labeling/pilot_compare.py prepare
    # Opus agent writes labels_opus.json
    python experiments/task2_aspect/labeling/pilot_compare.py check-labels
    # Haiku and Sonnet agents write review_haiku.json / review_sonnet.json
    python experiments/task2_aspect/labeling/pilot_compare.py compare

Everything is written to data/interim/task2_labeling/pilot_compare/ (data, not committed).
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labelkit.batches import SampleSentence, make_batches  # noqa: E402
from labelkit.outputs import (  # noqa: E402
    LABEL_SCHEMA,
    REVIEW_SCHEMA,
    LabelOutput,
    ReviewOutput,
    check_ids,
    dedupe_labels,
)
from labelkit.requests import Prompts, render_batch  # noqa: E402

from steam_review_translator.aspect.sentences import unescape_html  # noqa: E402
from steam_review_translator.core.config import get_path, load_yaml, project_root  # noqa: E402

N_REVIEWS = 20
REVIEWERS = ("haiku", "sonnet")
OUT = get_path("interim") / "task2_labeling" / "pilot_compare"


def _load_batches(cfg: dict) -> list:
    meta = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    chosen = set(meta["review_ids"])
    rows = [
        SampleSentence(**{k: r[k] for k in SampleSentence.__dataclass_fields__})
        for r in (
            json.loads(x)
            for x in (project_root() / cfg["input"]["sample"])
            .read_text(encoding="utf-8")
            .splitlines()
        )
        if r["review_id"] in chosen
    ]
    texts = (
        pl.scan_parquet(project_root() / cfg["input"]["reviews_parquet"])
        .filter(pl.col("review_id").is_in(sorted(chosen)))
        .select("review_id", "review_text")
        .collect()
    )
    review_texts = {r: unescape_html(t) for r, t in texts.iter_rows()}
    return make_batches(rows, review_texts, max_sentences=10_000)  # one block per game


def _render_all(batches: list, labels: dict | None = None) -> str:
    return "\n".join(render_batch(b, labels=labels) for b in batches)


def prepare(cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = [
        json.loads(x)
        for x in (project_root() / cfg["input"]["sample"]).read_text(encoding="utf-8").splitlines()
    ]
    train_reviews = sorted({r["review_id"] for r in sample if r["split"] == "train"})
    chosen = sorted(random.Random(cfg["seed"]).sample(train_reviews, N_REVIEWS))
    (OUT / "selection.json").write_text(
        json.dumps({"seed": cfg["seed"], "n_reviews": N_REVIEWS, "review_ids": chosen}, indent=2),
        encoding="utf-8",
    )
    batches = _load_batches(cfg)
    ids = [sid for b in batches for sid in b.sentence_ids]
    (OUT / "input.md").write_text(_render_all(batches), encoding="utf-8")
    (OUT / "label_schema.json").write_text(json.dumps(LABEL_SCHEMA, indent=2), encoding="utf-8")
    (OUT / "review_schema.json").write_text(json.dumps(REVIEW_SCHEMA, indent=2), encoding="utf-8")
    (OUT / "sentence_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    prompts = Prompts.load(project_root() / cfg["prompts_dir"])
    (OUT / "prompts_sha256.txt").write_text(prompts.sha256 + "\n", encoding="utf-8")
    games = sorted({b.game_name for b in batches})
    print(f"{N_REVIEWS} reviews, {len(ids)} sentences, {len(games)} games: {', '.join(games)}")
    print(f"wrote {OUT}")


def _labels() -> dict:
    out = LabelOutput.model_validate_json((OUT / "labels_opus.json").read_text(encoding="utf-8"))
    ids = json.loads((OUT / "sentence_ids.json").read_text(encoding="utf-8"))
    check_ids([s.sentence_id for s in out.sentences], ids)
    labels, fixed = {}, 0
    for s in out.sentences:
        labels[s.sentence_id], changed = dedupe_labels(s.labels)
        fixed += changed
    return {"labels": labels, "fixed": fixed}


def check_labels(cfg: dict) -> None:
    res = _labels()
    labels = res["labels"]
    n_lab = sum(bool(v) for v in labels.values())
    print(
        f"labels_opus.json valid: {len(labels)} sentences, {n_lab} with >= 1 label, "
        f"{res['fixed']} duplicate-aspect fixes"
    )
    (OUT / "review_input.md").write_text(_render_all(_load_batches(cfg), labels), encoding="utf-8")
    print(f"wrote {OUT / 'review_input.md'}")


def compare(cfg: dict) -> None:
    labels = _labels()["labels"]
    ids = json.loads((OUT / "sentence_ids.json").read_text(encoding="utf-8"))
    reviews = {}
    for name in REVIEWERS:
        out = ReviewOutput.model_validate_json(
            (OUT / f"review_{name}.json").read_text(encoding="utf-8")
        )
        check_ids([i.sentence_id for i in out.items], ids)
        reviews[name] = {i.sentence_id: i for i in out.items}
        flags = sum(i.verdict.value == "flag" for i in out.items)
        print(
            f"review_{name}.json valid: {flags} flagged / {len(ids)}, batch verdict "
            f"{out.batch_verdict.value} | {out.notes}"
        )
    sents = {
        s.sentence_id: (b.game_name, s.sentence)
        for b in _load_batches(cfg)
        for r in b.reviews
        for s in r.sentences
    }

    def fmt(labs: list) -> str:
        return "; ".join(f"{x.aspect.value}/{x.polarity.value}" for x in labs) or "[]"

    path = OUT / "compare.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "sentence_id",
                "game",
                "sentence",
                "opus_labels",
                *[
                    f"{n}_{c}"
                    for n in REVIEWERS
                    for c in ("verdict", "issue", "reasoning", "suggested")
                ],
                "my_correct_labels",
                "opus_correct (y/n)",
                "notes",
            ]
        )
        for sid in ids:
            row = [sid, *sents[sid], fmt(labels[sid])]
            for n in REVIEWERS:
                it = reviews[n][sid]
                row += [it.verdict.value, it.issue.value, it.reasoning, fmt(it.suggested_labels)]
            w.writerow([*row, "", "", ""])
    both = sum(
        reviews["haiku"][s].verdict.value == reviews["sonnet"][s].verdict.value == "flag"
        for s in ids
    )
    only_h = sum(
        reviews["haiku"][s].verdict.value == "flag" != reviews["sonnet"][s].verdict.value
        for s in ids
    )
    only_s = sum(
        reviews["sonnet"][s].verdict.value == "flag" != reviews["haiku"][s].verdict.value
        for s in ids
    )
    print(f"flag overlap: both {both}, only haiku {only_h}, only sonnet {only_s}")
    print(f"wrote {path}  <- open in Excel, fill my_correct_labels / opus_correct")


def main() -> int:
    cfg = load_yaml("configs/task2/labeling.yaml")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    actions = {"prepare": prepare, "check-labels": check_labels, "compare": compare}
    if cmd not in actions:
        print(__doc__)
        return 2
    actions[cmd](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
