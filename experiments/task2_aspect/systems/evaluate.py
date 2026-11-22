"""Score a registered Task 2 tagger on the labeled sentences of one split.

Usage (from the repo root):
    python experiments/task2_aspect/systems/evaluate.py <impl> <split> [key=value ...]
    e.g. evaluate.py mock val   |   evaluate.py peft test model_dir=artifacts/task2/peft

``impl`` is a name in ``ASPECT_REGISTRY``; ``key=value`` pairs are constructor params.
Writes artifacts/task2/eval/<impl>_<split>.json and prints the table.
"""

from __future__ import annotations

import json
import sys
import time

from steam_review_translator.aspect.metrics import format_score, score
from steam_review_translator.core.config import ComponentConfig, project_root
from steam_review_translator.core.interfaces import AspectTagger
from steam_review_translator.core.schemas import AspectLabel
from steam_review_translator.pipeline import ASPECT_REGISTRY, _build

LABELS = "data/processed/task2_labels.jsonl"


def load_split(split: str) -> tuple[list[str], list[list[AspectLabel]]]:
    """Sentences and gold labels of one split, in file order."""
    path = project_root() / LABELS
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == split]
    return [r["sentence"] for r in rows], [[AspectLabel(**x) for x in r["labels"]] for r in rows]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    impl, split = sys.argv[1], sys.argv[2]
    params = dict(a.split("=", 1) for a in sys.argv[3:])
    tagger = _build("aspect", ASPECT_REGISTRY, ComponentConfig(impl=impl, params=params),
                    AspectTagger)  # fmt: skip
    sentences, gold = load_split(split)
    start = time.perf_counter()
    pred = tagger.tag(sentences)
    seconds = time.perf_counter() - start
    s = score(gold, pred)
    s.update(impl=impl, split=split, params=params, seconds=round(seconds, 1))
    out = project_root() / "artifacts/task2/eval" / f"{impl}_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(format_score(s, f"{impl} on {split} ({seconds:.1f}s)"))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
