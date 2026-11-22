"""BEAM description of the Steam Review Translator: one node list -> Turtle + diagram.

Uses the same vocabulary as the group's design document (beam:Data, beam:Symbol,
beam:StatisticalModel, beam:Training, beam:Inference, beam:Transformation, beam:Engineering,
beam:Actor, beam:Container; beam:flowsTo, beam:partOf).

Usage (from the repo root):  python paper/beam/make_beam.py
Writes paper/beam/beam.ttl and paper/beam/beam_pipeline.png (+ .pdf).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

HERE = Path(__file__).resolve().parent

CONTAINERS = {
    "Task1": "Task 1 -- Batch Sentiment Aggregation",
    "Task2": "Task 2 -- Aspect-Based Opinion Mining (sentence-level ABSA)",
    "Task3": "Task 3 -- Meta-Review Generation (multi-document summarization)",
}

# id: (BEAM type, label, containers, (x, y))
NODES: dict[str, tuple[str, str, list[str], tuple[float, float]]] = {
    # shared input
    "batchReviews": (
        "Data",
        "Batch of reviews of one game (input)",
        ["Task1", "Task2", "Task3"],
        (1.3, 6.4),
    ),
    # Task 1
    "t1_trainData": (
        "Data",
        "Metacritic user reviews (train), label from user score",
        ["Task1"],
        (4.0, 11.1),
    ),
    "t1_training": ("Training", "Fine-tuning", ["Task1"], (7.0, 11.1)),
    "t1_model": ("StatisticalModel", "RoBERTa-base (S4)", ["Task1"], (10.0, 11.1)),
    "t1_inference": ("Inference", "Inference (per review)", ["Task1"], (10.0, 9.6)),
    "t1_labels": ("Symbol", "Sentiment label per review (pos / neg / neu)", ["Task1"], (12.8, 9.6)),
    "t1_aggregate": ("Transformation", "Aggregate (class shares)", ["Task1"], (15.4, 9.6)),
    "t1_distribution": (
        "Symbol",
        "Sentiment distribution + dominant label (output)",
        ["Task1"],
        (18.0, 9.6),
    ),
    # Task 2
    "t2_sample": (
        "Data",
        "Sampled Steam review sentences (7,002; 13 games)",
        ["Task2"],
        (4.0, 7.8),
    ),
    "t2_annotator": ("Actor", "LLM annotator (Claude) + team guideline", ["Task2"], (6.4, 6.55)),
    "t2_annotation": ("Engineering", "Annotation (aspect, polarity)", ["Task2"], (7.0, 7.8)),
    "t2_labeled": (
        "Data",
        "Labeled sentences (train / val / test by game)",
        ["Task2"],
        (10.0, 7.8),
    ),
    "t2_training": ("Training", "LoRA fine-tuning", ["Task2"], (12.8, 7.8)),
    "t2_model": (
        "StatisticalModel",
        "RoBERTa-base + LoRA, 11 aspect heads",
        ["Task2"],
        (15.4, 7.8),
    ),
    "t2_split": ("Transformation", "Sentence splitting (spaCy)", ["Task2"], (4.0, 5.3)),
    "t2_sentences": ("Data", "Review sentences", ["Task2"], (7.0, 5.3)),
    "t2_inference": ("Inference", "Inference (per sentence)", ["Task2"], (10.0, 5.3)),
    "t2_pairs": ("Symbol", "(aspect, polarity) pairs per sentence", ["Task2"], (12.8, 5.3)),
    "t2_aggregate": (
        "Transformation",
        "Aggregate (frequency, majority polarity)",
        ["Task2"],
        (15.4, 5.3),
    ),
    "aspectTable": (
        "Symbol",
        "Aspect table: frequency + polarity per aspect (output)",
        ["Task2", "Task3"],
        (18.0, 5.3),
    ),
    # Task 3
    "t3_gameWikiSum": (
        "Data",
        "GameWikiSum: game reviews + Wikipedia summaries",
        ["Task3"],
        (4.0, 3.1),
    ),
    "t3_prepTrain": (
        "Transformation",
        "Paragraph selection + pseudo aspect table (lexicon) + serialization",
        ["Task3"],
        (7.0, 3.1),
    ),
    "t3_trainInputs": (
        "Data",
        "Serialized inputs: aspect prefix + text -> summary",
        ["Task3"],
        (10.0, 3.1),
    ),
    "t3_training": ("Training", "Fine-tuning", ["Task3"], (12.8, 3.1)),
    "t3_model": ("StatisticalModel", "BART-base + aspect prefix (E2)", ["Task3"], (15.4, 3.1)),
    "t3_prepInput": (
        "Transformation",
        "Paragraph selection + serialization",
        ["Task3"],
        (7.0, 1.0),
    ),
    "t3_input": ("Data", "Aspect prefix + selected review text", ["Task3"], (10.0, 1.0)),
    "t3_inference": ("Inference", "Generation (beam search)", ["Task3"], (15.4, 1.0)),
    "t3_metaReview": ("Data", "Meta-review (output)", ["Task3"], (18.0, 1.0)),
}

EDGES = [
    ("t1_trainData", "t1_training"), ("t1_training", "t1_model"),
    ("t1_model", "t1_inference"), ("batchReviews", "t1_inference"),
    ("t1_inference", "t1_labels"), ("t1_labels", "t1_aggregate"),
    ("t1_aggregate", "t1_distribution"),
    ("t2_sample", "t2_annotation"), ("t2_annotator", "t2_annotation"),
    ("t2_annotation", "t2_labeled"), ("t2_labeled", "t2_training"),
    ("t2_training", "t2_model"),
    ("batchReviews", "t2_split"), ("t2_split", "t2_sentences"),
    ("t2_sentences", "t2_inference"), ("t2_model", "t2_inference"),
    ("t2_inference", "t2_pairs"), ("t2_pairs", "t2_aggregate"),
    ("t2_aggregate", "aspectTable"),
    ("t3_gameWikiSum", "t3_prepTrain"), ("t3_prepTrain", "t3_trainInputs"),
    ("t3_trainInputs", "t3_training"), ("t3_training", "t3_model"),
    ("batchReviews", "t3_prepInput"), ("aspectTable", "t3_prepInput"),
    ("t3_prepInput", "t3_input"), ("t3_input", "t3_inference"),
    ("t3_model", "t3_inference"), ("t3_inference", "t3_metaReview"),
]  # fmt: skip

# Long edges drawn as orthogonal polylines (waypoints) so they do not cross other nodes.
ROUTES = {
    ("batchReviews", "t1_inference"): [(2.475, 6.75), (2.72, 6.75), (2.72, 9.6), (8.825, 9.6)],
    ("batchReviews", "t3_prepInput"): [(2.475, 6.05), (2.72, 6.05), (2.72, 1.0), (5.825, 1.0)],
    ("aspectTable", "t3_prepInput"): [(19.175, 5.3), (19.5, 5.3), (19.5, 0.3), (7.0, 0.3),
                                      (7.0, 0.45)],
}  # fmt: skip

CONTAINER_BOX = {"Task1": (2.9, 8.85, 19.7, 12.0), "Task2": (2.9, 4.5, 19.7, 8.75),
                 "Task3": (2.9, 0.15, 19.7, 4.4)}  # fmt: skip


def write_turtle(path: Path) -> None:
    out = [
        "# BEAM description of the Steam Review Translator (generated by make_beam.py).",
        "# The beam: namespace below is a placeholder: replace it with the one from the course.",
        "@prefix beam: <https://semsys.ai.wu.ac.at/beam/beam_core.ttl#> .",
        "@prefix ex:   <https://example.org/steam-review-translator#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "",
        "ex:SteamReviewTranslator a beam:SystemDescription ;",
        '    rdfs:label "Steam Review Translator" ;',
        '    rdfs:comment "Turns a batch of reviews of one game into a sentiment distribution, '
        'an aspect table and a meta-review." .',
        "",
    ]
    for cid, label in CONTAINERS.items():
        out += [f"ex:{cid} a beam:Container ;", f'    rdfs:label "{label}" ;',
                "    beam:partOf ex:SteamReviewTranslator .", ""]  # fmt: skip
    outgoing: dict[str, list[str]] = {}
    for a, b in EDGES:
        outgoing.setdefault(a, []).append(b)
    for nid, (kind, label, parts, _) in NODES.items():
        lines = [f"ex:{nid} a beam:{kind} ;", f'    rdfs:label "{label}" ;',
                 "    beam:partOf " + ", ".join(f"ex:{p}" for p in parts)]  # fmt: skip
        if nid in outgoing:
            lines[-1] += " ;"
            lines.append("    beam:flowsTo " + ", ".join(f"ex:{t}" for t in outgoing[nid]))
        lines[-1] += " ."
        out += lines + [""]
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")


STYLE = {
    "Data": {"fc": "#dbeafe", "ec": "#1d4ed8"},
    "Symbol": {"fc": "#dcfce7", "ec": "#15803d"},
    "StatisticalModel": {"fc": "#fde68a", "ec": "#b45309"},
    "Training": {"fc": "#fce7f3", "ec": "#be185d"},
    "Inference": {"fc": "#fce7f3", "ec": "#be185d"},
    "Transformation": {"fc": "#ede9fe", "ec": "#6d28d9"},
    "Engineering": {"fc": "#ede9fe", "ec": "#6d28d9"},
    "Actor": {"fc": "#e5e7eb", "ec": "#374151"},
}
W, H = 2.35, 1.1  # node box size


def shape(ax, kind: str, x: float, y: float):
    s = STYLE[kind]
    kw = {"fc": s["fc"], "ec": s["ec"], "lw": 1.3, "zorder": 3}
    if kind in ("Data", "Symbol"):
        p = Rectangle((x - W / 2, y - H / 2), W, H, **kw)
    elif kind == "StatisticalModel":
        hw, hh, cut = W / 2, H / 2, 0.3  # wide hexagon
        p = Polygon([(x - hw, y), (x - hw + cut, y + hh), (x + hw - cut, y + hh), (x + hw, y),
                     (x + hw - cut, y - hh), (x - hw + cut, y - hh)], **kw)  # fmt: skip
    elif kind == "Actor":
        p = Polygon([(x, y + H / 2), (x - W / 2.4, y - H / 2), (x + W / 2.4, y - H / 2)], **kw)
    else:
        p = Ellipse((x, y), W, H, **kw)
    ax.add_patch(p)
    return p


def draw(png: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 12.4))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12.3)
    ax.axis("off")
    for cid, (x0, y0, x1, y1) in CONTAINER_BOX.items():
        box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02",
                             fc="none", ec="#6b7280", lw=1.2, ls="--", zorder=1)  # fmt: skip
        ax.add_patch(box)
        ax.text(x0 + 0.12, y1 - 0.12, CONTAINERS[cid], ha="left", va="top", fontsize=11,
                fontweight="bold", color="#374151", zorder=2)  # fmt: skip
    patches = {}
    for nid, (kind, label, _, (x, y)) in NODES.items():
        patches[nid] = shape(ax, kind, x, y)
        if kind == "Actor":  # label to the right of the triangle
            lx = x + W / 2.4 + 0.12
            ax.text(lx, y + 0.25, kind, ha="left", va="center", fontsize=7.5, style="italic",
                    color=STYLE[kind]["ec"], zorder=4)  # fmt: skip
            ax.text(lx, y - 0.12, "\n".join(textwrap.wrap(label, 22)), ha="left",
                    va="center", fontsize=8.3, zorder=4)  # fmt: skip
            continue
        ax.text(x, y + H / 2 - 0.16, kind, ha="center", va="center", fontsize=7.5,
                style="italic", color=STYLE[kind]["ec"], zorder=4)  # fmt: skip
        ax.text(x, y - 0.1, "\n".join(textwrap.wrap(label, 24)), ha="center", va="center",
                fontsize=8.3, zorder=4)  # fmt: skip
    for a, b in EDGES:
        if (a, b) in ROUTES:
            pts = ROUTES[(a, b)]
            xs, ys = zip(*pts[:-1], strict=True)
            ax.plot(xs, ys, color="#374151", lw=1.1, zorder=2)
            ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=13,
                                         lw=1.1, color="#374151", zorder=2))  # fmt: skip
            continue
        ax.add_patch(FancyArrowPatch(patchA=patches[a], patchB=patches[b],
                                     posA=NODES[a][3], posB=NODES[b][3],
                                     arrowstyle="-|>", mutation_scale=13, lw=1.1,
                                     color="#374151", zorder=2,
                                     connectionstyle="arc3,rad=0.0"))  # fmt: skip
    legend = [("Data", "Data"), ("Symbol", "Symbol"), ("StatisticalModel", "Model"),
              ("Training", "Process"), ("Actor", "Actor")]  # fmt: skip
    for i, (kind, name) in enumerate(legend):
        lx = 0.35
        ly = 11.5 - i * 0.75
        ax.add_patch(Rectangle((lx, ly - 0.2), 0.45, 0.4, fc=STYLE[kind]["fc"],
                               ec=STYLE[kind]["ec"], lw=1.2))  # fmt: skip
        ax.text(lx + 0.6, ly, name, va="center", fontsize=9)
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    write_turtle(HERE / "beam.ttl")
    draw(HERE / "beam_pipeline.png")
    print("wrote", HERE / "beam.ttl", "and", HERE / "beam_pipeline.png")
