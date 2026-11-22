"""Sentence-level scores of an ``AspectTagger`` against gold labels.

Every sentence has a set of ``(aspect, polarity)`` pairs. Reported:
- ``pair``: micro P/R/F1 over (aspect, polarity) pairs, the main score;
- ``aspect``: micro P/R/F1 over aspects only (detection, polarity ignored);
- ``per_aspect``: P/R/F1/support of aspect detection for each aspect, and their macro F1;
- ``polarity_acc``: polarity accuracy on aspects found in both gold and prediction;
- ``exact``: share of sentences whose pair set matches exactly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from steam_review_translator.core.schemas import Aspect, AspectLabel


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def score(
    gold: Sequence[Sequence[AspectLabel]], pred: Sequence[Sequence[AspectLabel]]
) -> dict[str, Any]:
    """Score ``pred`` against ``gold`` (one label list per sentence, same order)."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold vs {len(pred)} predicted sentences")
    pair: Counter = Counter()
    asp: Counter = Counter()
    per: dict[Aspect, Counter] = {a: Counter() for a in Aspect}
    pol_same = pol_total = exact = 0
    for g_labels, p_labels in zip(gold, pred, strict=True):
        g = {(x.aspect, x.polarity) for x in g_labels}
        p = {(x.aspect, x.polarity) for x in p_labels}
        pair.update(tp=len(g & p), fp=len(p - g), fn=len(g - p))
        ga, pa = dict(g), dict(p)
        asp.update(tp=len(ga.keys() & pa.keys()), fp=len(pa.keys() - ga.keys()),
                   fn=len(ga.keys() - pa.keys()))  # fmt: skip
        for a in ga.keys() | pa.keys():
            per[a].update(tp=a in ga and a in pa, fp=a not in ga, fn=a not in pa)
        for a in ga.keys() & pa.keys():
            pol_total += 1
            pol_same += ga[a] == pa[a]
        exact += g == p
    per_aspect = {
        a.value: {**_prf(c["tp"], c["fp"], c["fn"]), "support": c["tp"] + c["fn"]}
        for a, c in per.items()
    }
    present = [v["f1"] for v in per_aspect.values() if v["support"]]
    return {
        "n_sentences": len(gold),
        "pair": _prf(pair["tp"], pair["fp"], pair["fn"]),
        "aspect": _prf(asp["tp"], asp["fp"], asp["fn"]),
        "aspect_macro_f1": sum(present) / len(present) if present else 0.0,
        "polarity_acc": pol_same / pol_total if pol_total else 0.0,
        "exact": exact / len(gold) if gold else 0.0,
        "per_aspect": per_aspect,
    }


def format_score(s: dict[str, Any], title: str = "") -> str:
    """A plain-text table of ``score`` output."""
    lines = [title] if title else []
    pr, ar = s["pair"], s["aspect"]
    lines.append(
        f"pair    P {pr['p']:.3f}  R {pr['r']:.3f}  F1 {pr['f1']:.3f}   |  "
        f"aspect P {ar['p']:.3f}  R {ar['r']:.3f}  F1 {ar['f1']:.3f}"
    )
    lines.append(
        f"aspect macro F1 {s['aspect_macro_f1']:.3f}  polarity acc {s['polarity_acc']:.3f}  "
        f"exact {s['exact']:.3f}  ({s['n_sentences']} sentences)"
    )
    lines.append(f"{'aspect':<12} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    rows = sorted(s["per_aspect"].items(), key=lambda kv: -kv[1]["support"])
    for name, v in rows:
        lines.append(f"{name:<12} {v['p']:>6.3f} {v['r']:>6.3f} {v['f1']:>6.3f} {v['support']:>8}")
    return "\n".join(lines)
