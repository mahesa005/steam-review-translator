"""Full labeling run with Claude Code agents instead of the Batch API (see DESIGN.md).

Per batch: Opus labels -> Sonnet and Opus review independently -> Opus decides flagged
sentences. Agents only read input files and write one JSON file each; this script prepares
every input and validates every output with labelkit.

Usage (from the repo root):
    python experiments/task2_aspect/labeling/run_agents.py prepare
    python experiments/task2_aspect/labeling/run_agents.py step <batch_id>   # after each agent
    python experiments/task2_aspect/labeling/run_agents.py status
    python experiments/task2_aspect/labeling/run_agents.py merge

Files per batch in <run_dir>/<batch_id>/:
    label_input.md -> labels.json -> review_input.md -> review_<reviewer>.json
    -> finalize_input.md -> finalize.json (only if something was flagged) -> final.json
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labelkit.batches import Batch, SampleSentence, make_batches  # noqa: E402
from labelkit.outputs import (  # noqa: E402
    FINALIZE_SCHEMA,
    LABEL_SCHEMA,
    REVIEW_SCHEMA,
    FinalizeOutput,
    Label,
    LabelOutput,
    ReviewItem,
    ReviewOutput,
    check_ids,
    dedupe_labels,
)
from labelkit.requests import Prompts, render_batch  # noqa: E402

from steam_review_translator.aspect.sentences import unescape_html  # noqa: E402
from steam_review_translator.core.config import load_yaml, project_root  # noqa: E402


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _fmt(labels: list[Label]) -> str:
    return "; ".join(f"{x.aspect.value}/{x.polarity.value}" for x in labels) or "[]"


def _dump(labels: list[Label]) -> list[dict[str, str]]:
    return [{"aspect": x.aspect.value, "polarity": x.polarity.value} for x in labels]


class Run:
    """Paths and batches of one agent run."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.root = project_root() / cfg["agents"]["run_dir"]
        self.reviewers: list[str] = list(cfg["agents"]["reviewers"])

    def batches(self) -> list[Batch]:
        rows_raw = [
            json.loads(x) for x in _read(project_root() / self.cfg["input"]["sample"]).splitlines()
        ]
        rows = [
            SampleSentence(**{k: r[k] for k in SampleSentence.__dataclass_fields__})
            for r in rows_raw
        ]
        ids = sorted({r.review_id for r in rows})
        texts = (
            pl.scan_parquet(project_root() / self.cfg["input"]["reviews_parquet"])
            .filter(pl.col("review_id").is_in(ids))
            .select("review_id", "review_text")
            .collect()
        )
        review_texts = {r: unescape_html(t) for r, t in texts.iter_rows()}
        return make_batches(rows, review_texts, self.cfg["agents"]["max_sentences"])

    def batch(self, batch_id: str) -> Batch:
        for b in self.batches():
            if b.batch_id == batch_id:
                return b
        raise SystemExit(f"unknown batch {batch_id}")

    def dir(self, batch_id: str) -> Path:
        return self.root / batch_id

    # -- stage outputs, validated ------------------------------------------------------
    def labels(self, b: Batch) -> dict[str, list[Label]]:
        out = LabelOutput.model_validate_json(_read(self.dir(b.batch_id) / "labels.json"))
        check_ids([s.sentence_id for s in out.sentences], b.sentence_ids)
        return {s.sentence_id: dedupe_labels(s.labels)[0] for s in out.sentences}

    def reviews(self, b: Batch) -> dict[str, dict[str, ReviewItem]]:
        res = {}
        for name in self.reviewers:
            out = ReviewOutput.model_validate_json(
                _read(self.dir(b.batch_id) / f"review_{name}.json")
            )
            check_ids([i.sentence_id for i in out.items], b.sentence_ids)
            res[name] = {i.sentence_id: i for i in out.items}
        return res

    def flagged(self, b: Batch, reviews: dict[str, dict[str, ReviewItem]]) -> list[str]:
        return [
            sid
            for sid in b.sentence_ids
            if any(reviews[n][sid].verdict.value == "flag" for n in self.reviewers)
        ]


def prepare(run: Run) -> None:
    run.root.mkdir(parents=True, exist_ok=True)
    batches = run.batches()
    _write(run.root / "label_schema.json", json.dumps(LABEL_SCHEMA, indent=2))
    _write(run.root / "review_schema.json", json.dumps(REVIEW_SCHEMA, indent=2))
    _write(run.root / "finalize_schema.json", json.dumps(FINALIZE_SCHEMA, indent=2))
    prompts = Prompts.load(project_root() / run.cfg["prompts_dir"])
    manifest = {
        "prompts_sha256": prompts.sha256,
        "agents": run.cfg["agents"],
        "batches": [
            {
                "batch_id": b.batch_id,
                "game": b.game_name,
                "split": b.split,
                "n_sentences": b.n_sentences,
            }
            for b in batches
        ],
    }
    _write(run.root / "manifest.json", json.dumps(manifest, indent=2))
    for b in batches:
        d = run.dir(b.batch_id)
        d.mkdir(exist_ok=True)
        _write(d / "label_input.md", render_batch(b))
    n = sum(b.n_sentences for b in batches)
    print(f"{len(batches)} batches, {n} sentences -> {run.root}")


def stage_of(run: Run, b: Batch) -> str:
    """The next thing a batch needs: label, review, finalize or done."""
    d = run.dir(b.batch_id)
    if (d / "final.json").exists():
        return "done"
    if not (d / "labels.json").exists():
        return "label"
    if not all((d / f"review_{n}.json").exists() for n in run.reviewers):
        return "review"
    return "finalize"


def step(run: Run, batch_id: str) -> None:
    """Validate what exists for one batch and write the next stage's input."""
    b = run.batch(batch_id)
    d = run.dir(batch_id)
    stage = stage_of(run, b)
    if stage == "label":
        print(f"{batch_id}: needs labels.json")
        return
    labels = run.labels(b)
    if not (d / "review_input.md").exists():
        _write(d / "review_input.md", render_batch(b, labels=labels))
    if stage == "review":
        missing = [n for n in run.reviewers if not (d / f"review_{n}.json").exists()]
        print(f"{batch_id}: labels ok ({len(labels)}); needs review from {', '.join(missing)}")
        return
    if stage == "done":
        print(f"{batch_id}: done")
        return
    reviews = run.reviews(b)
    flagged = run.flagged(b, reviews)
    decisions: dict = {}
    if flagged:
        if not (d / "finalize.json").exists():
            per_sentence = {sid: [reviews[n][sid] for n in run.reviewers] for sid in flagged}
            _write(
                d / "finalize_input.md",
                render_batch(b, labels=labels, reviews=per_sentence, only_ids=set(flagged)),
            )
            _write(d / "flagged_ids.json", json.dumps(flagged))
            print(f"{batch_id}: reviews ok; {len(flagged)} flagged, needs finalize.json")
            return
        out = FinalizeOutput.model_validate_json(_read(d / "finalize.json"))
        check_ids([x.sentence_id for x in out.decisions], flagged)
        decisions = {x.sentence_id: x for x in out.decisions}
    final = []
    for sid in b.sentence_ids:
        rec: dict = {
            "sentence_id": sid,
            "labels": _dump(labels[sid]),
            "initial_labels": _dump(labels[sid]),
            "flagged_by": [n for n in run.reviewers if reviews[n][sid].verdict.value == "flag"],
            "decision": None,
            "rationale": "",
        }
        if sid in decisions:
            x = decisions[sid]
            rec["labels"] = _dump(dedupe_labels(x.final_labels)[0])
            rec["decision"] = x.decision.value
            rec["rationale"] = x.rationale
        final.append(rec)
    _write(d / "final.json", json.dumps(final, indent=1, ensure_ascii=False))
    changed = sum(r["labels"] != r["initial_labels"] for r in final)
    print(f"{batch_id}: done; {len(flagged)} flagged, {changed} changed")


def status(run: Run) -> None:
    counts: dict[str, int] = {}
    rows = []
    for b in run.batches():
        s = stage_of(run, b)
        counts[s] = counts.get(s, 0) + 1
        rows.append(f"{b.batch_id:>14}  {b.split:5}  {b.n_sentences:4}  {s}")
    print("\n".join(rows))
    print(counts)


def merge(run: Run) -> None:
    batches = run.batches()
    pending = [b.batch_id for b in batches if stage_of(run, b) != "done"]
    if pending:
        raise SystemExit(f"{len(pending)} batches not done, e.g. {pending[:5]}")
    meta = {s.sentence_id: (b, r, s) for b in batches for r in b.reviews for s in r.sentences}
    records = []
    for b in batches:
        for rec in json.loads(_read(run.dir(b.batch_id) / "final.json")):
            _, review, s = meta[rec["sentence_id"]]
            records.append(
                {
                    "sentence_id": s.sentence_id,
                    "review_id": review.review_id,
                    "game_id": b.game_id,
                    "game": b.game_name,
                    "split": b.split,
                    "sentence": s.sentence,
                    **rec,
                }
            )
    out = project_root() / run.cfg["output"]["final_labels"]
    _write(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    # verification queue: every flagged sentence + a random share of unflagged, per split
    rng = random.Random(run.cfg["seed"])
    share = run.cfg["verification"]["random_unflagged_share"]
    queue = [(r, "flagged") for r in records if r["flagged_by"]]
    for split in sorted({r["split"] for r in records}):
        clean = [r for r in records if r["split"] == split and not r["flagged_by"]]
        queue += [(r, "random") for r in rng.sample(clean, round(len(clean) * share))]
    qpath = project_root() / run.cfg["output"]["verify_queue"]
    with qpath.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["sentence_id", "split", "game", "reason", "sentence", "initial_labels",
             "final_labels", "flagged_by", "rationale", "my_correct_labels", "notes"]
        )  # fmt: skip
        for r, why in queue:
            fmt = "; ".join(f"{x['aspect']}/{x['polarity']}" for x in r["labels"]) or "[]"
            ini = "; ".join(f"{x['aspect']}/{x['polarity']}" for x in r["initial_labels"]) or "[]"
            w.writerow(
                [r["sentence_id"], r["split"], r["game"], why, r["sentence"], ini, fmt,
                 ",".join(r["flagged_by"]), r["rationale"], "", ""]
            )  # fmt: skip
    n_flag = sum(1 for _, why in queue if why == "flagged")
    print(f"{len(records)} sentences -> {out}")
    n_rand = len(queue) - n_flag
    print(f"verify queue: {len(queue)} ({n_flag} flagged + {n_rand} random) -> {qpath}")


def main() -> int:
    run = Run(load_yaml("configs/task2/labeling.yaml"))
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "step" and len(sys.argv) == 3:
        step(run, sys.argv[2])
    elif cmd in {"prepare", "status", "merge"}:
        {"prepare": prepare, "status": status, "merge": merge}[cmd](run)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
