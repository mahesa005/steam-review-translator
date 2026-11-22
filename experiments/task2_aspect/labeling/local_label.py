"""Label sentences with a local model through Ollama, one review per request.

Calibration first: the model labels the reviews whose batches the Claude agents already
finished (run_agents.py), and ``report`` compares its labels with those. Same guideline and
label prompt as the agents, same output schema and validation (labelkit).

Usage (from the repo root, with ``ollama serve`` running):
    python experiments/task2_aspect/labeling/local_label.py calibrate [--limit N]
    python experiments/task2_aspect/labeling/local_label.py report

Output: <local.out_dir>/<model>.jsonl, one line per review; reruns skip reviews already there.
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labelkit.batches import Batch  # noqa: E402
from labelkit.outputs import LABEL_SCHEMA, LabelOutput, check_ids, dedupe_labels  # noqa: E402
from labelkit.requests import Prompts, render_batch  # noqa: E402
from run_agents import Run, stage_of  # noqa: E402

from steam_review_translator.core.config import load_yaml, project_root  # noqa: E402

Pair = tuple[str, str]


def _out_path(cfg: dict) -> Path:
    local = cfg["local"]
    n = local["samples"]
    name = local["model"].replace(":", "_") + (f"_x{n}" if n > 1 else "")
    return project_root() / local["out_dir"] / f"{name}.jsonl"


def _done_batches(run: Run) -> list[Batch]:
    return [b for b in run.batches() if stage_of(run, b) == "done"]


def _reference(run: Run, batches: list[Batch]) -> dict[str, set[Pair]]:
    """Final agent labels per sentence id, as (aspect, polarity) sets."""
    ref: dict[str, set[Pair]] = {}
    for b in batches:
        for rec in json.loads((run.dir(b.batch_id) / "final.json").read_text(encoding="utf-8")):
            ref[rec["sentence_id"]] = {(x["aspect"], x["polarity"]) for x in rec["labels"]}
    return ref


def _chat(cfg: dict, system: str, user: str, seed: int) -> tuple[str, int]:
    local = cfg["local"]
    temperature = local["temperature"] if local["samples"] > 1 else 0
    body = {
        "model": local["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": LABEL_SCHEMA,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": local["num_ctx"]},
    }
    req = urllib.request.Request(
        f"{local['host']}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = json.loads(resp.read())
    return out["message"]["content"], out.get("prompt_eval_count", 0)


def _one_sample(cfg: dict, system: str, one: Batch, seed: int) -> tuple[dict, str]:
    """One answer for a single-review batch; one retry on invalid output."""
    error = ""
    for attempt in range(2):
        try:
            content, _ = _chat(cfg, system, render_batch(one), seed + 1000 * attempt)
            out = LabelOutput.model_validate_json(content)
            check_ids([s.sentence_id for s in out.sentences], one.sentence_ids)
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            error = str(exc)[:300]
            continue
        labels = {
            s.sentence_id: [x.model_dump(mode="json") for x in dedupe_labels(s.labels)[0]]
            for s in out.sentences
        }
        return labels, ""
    return {}, error


def label_review(cfg: dict, system: str, one: Batch) -> dict:
    """``local.samples`` independent answers for one review (failed ones are left out)."""
    start = time.perf_counter()
    samples, errors = [], []
    for i in range(cfg["local"]["samples"]):
        labels, error = _one_sample(cfg, system, one, cfg["seed"] + i)
        if error:
            errors.append(error)
        else:
            samples.append(labels)
    return {
        "review_id": one.reviews[0].review_id,
        "samples": samples,
        "error": "" if samples else errors[-1],
        "n_failed": len(errors),
        "seconds": round(time.perf_counter() - start, 1),
    }


def vote(samples: list[dict], min_votes: int) -> dict[str, set[Pair]]:
    """Keep an aspect named in >= ``min_votes`` samples, with its most frequent polarity."""
    out: dict[str, set[Pair]] = {}
    for sid in samples[0]:
        aspects: dict[str, Counter] = {}
        for s in samples:
            for x in s[sid]:
                aspects.setdefault(x["aspect"], Counter())[x["polarity"]] += 1
        out[sid] = {
            (a, pols.most_common(1)[0][0])
            for a, pols in aspects.items()
            if sum(pols.values()) >= min_votes
        }
    return out


def calibrate(cfg: dict, limit: int | None) -> None:
    run = Run(cfg)
    prompts = Prompts.load(project_root() / cfg["prompts_dir"])
    system = prompts.guideline + "\n\n" + prompts.stages["label"]
    singles = [
        Batch(str(r.review_id), b.game_id, b.game_name, b.split, [r])
        for b in _done_batches(run)
        for r in b.reviews
    ]
    random.Random(cfg["seed"]).shuffle(singles)  # a --limit subset spans every game
    if limit is not None:
        singles = singles[:limit]
    out = _out_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        seen = {json.loads(x)["review_id"] for x in out.read_text(encoding="utf-8").splitlines()}
    todo = [s for s in singles if s.reviews[0].review_id not in seen]
    print(f"{len(singles)} reviews selected, {len(todo)} to label -> {out}")
    for k, one in enumerate(todo, 1):
        rec = label_review(cfg, system, one)
        with out.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = f"ERROR {rec['error'][:80]}" if rec["error"] else "ok"
        size = f"{one.n_sentences} sent"
        print(f"[{k}/{len(todo)}] {one.batch_id} ({size}) {rec['seconds']}s {status}")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _score(ref: dict[str, set[Pair]], pred: dict[str, set[Pair]], detail: bool) -> float:
    """Print agreement of ``pred`` with ``ref`` on their shared sentences; return pair F1."""
    ids = [sid for sid in pred if sid in ref]
    pair: Counter = Counter()
    asp: dict[str, Counter] = {}
    pol_same = pol_total = exact = 0
    empty: Counter = Counter()
    misses: Counter = Counter()
    for sid in ids:
        r, p = ref[sid], pred[sid]
        pair.update(tp=len(r & p), fp=len(p - r), fn=len(r - p))
        ra, pa = {a for a, _ in r}, {a for a, _ in p}
        for a in ra | pa:
            asp.setdefault(a, Counter()).update(tp=a in ra and a in pa, fp=a in pa - ra,
                                                fn=a in ra - pa)  # fmt: skip
        rpol, ppol = dict(r), dict(p)
        for a in ra & pa:
            pol_total += 1
            pol_same += rpol[a] == ppol[a]
        exact += r == p
        empty[(not r, not p)] += 1
        misses.update(f"missed {a}" for a in ra - pa)
        misses.update(f"extra {a}" for a in pa - ra)

    n = max(len(ids), 1)
    pp, pr, pf = _prf(pair["tp"], pair["fp"], pair["fn"])
    pol = pol_same / max(pol_total, 1)
    print(f"  pairs P {pp:.3f}  R {pr:.3f}  F1 {pf:.3f} | exact {exact / n:.3f} | "
          f"polarity acc {pol:.3f} | only-local [] {empty[(False, True)]}")  # fmt: skip
    if not detail:
        return pf
    print(f"  {len(ids)} sentences; empty []: both {empty[(True, True)]}, "
          f"only Opus {empty[(True, False)]}, neither {empty[(False, False)]}")  # fmt: skip
    print(f"\n  {'aspect':<12} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    for a in sorted(asp, key=lambda a: -(asp[a]["tp"] + asp[a]["fn"])):
        c = asp[a]
        ap, ar, af = _prf(c["tp"], c["fp"], c["fn"])
        print(f"  {a:<12} {ap:>6.3f} {ar:>6.3f} {af:>6.3f} {c['tp'] + c['fn']:>8}")
    print("\n  most common differences:")
    for what, k in misses.most_common(10):
        print(f"    {what:<22} {k}")
    return pf


def report(cfg: dict) -> None:
    run = Run(cfg)
    ref = _reference(run, _done_batches(run))
    recs = [json.loads(x) for x in _out_path(cfg).read_text(encoding="utf-8").splitlines()]
    ok = [r for r in recs if not r["error"]]
    seconds = sum(r["seconds"] for r in recs)
    n_sent = sum(len(r["samples"][0]) for r in ok)
    n = cfg["local"]["samples"]
    print(f"model {cfg['local']['model']} x{n}: {len(recs)} reviews, "
          f"{len(recs) - len(ok)} failed, {sum(r['n_failed'] for r in recs)} failed samples, "
          f"{seconds / 60:.1f} min ({seconds / max(n_sent, 1):.2f} s/sentence)")  # fmt: skip
    scores = {}
    for k in range(1, n + 1):
        pred = {sid: labs for r in ok for sid, labs in vote(r["samples"], k).items()}
        print(f"\nvote >= {k} of {n}:")
        scores[k] = _score(ref, pred, detail=False)
    best = max(scores, key=lambda k: scores[k])
    print(f"\nbest: vote >= {best} of {n}")
    _score(ref, {sid: labs for r in ok for sid, labs in vote(r["samples"], best).items()}, True)
    for r in [r for r in recs if r["error"]][:5]:
        print(f"failed review {r['review_id']}: {r['error'][:150]}")


def main() -> int:
    cfg = load_yaml("configs/task2/labeling.yaml")
    args = sys.argv[1:]
    if args[:1] == ["calibrate"]:
        limit = int(args[2]) if args[1:2] == ["--limit"] and len(args) == 3 else None
        calibrate(cfg, limit)
    elif args == ["report"]:
        report(cfg)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
