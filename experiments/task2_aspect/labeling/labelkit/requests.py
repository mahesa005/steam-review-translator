"""Render the prompts and build Message Batches API request params for each stage.

Request layout (identical prefix within a stage, so it is cached):
    system = [guideline.md, <stage>.md  (cache breakpoint)]
    user   = the batch: game name, full reviews, sentence ids (+ labels / verdicts)
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from labelkit.batches import Batch
from labelkit.outputs import FINALIZE_SCHEMA, LABEL_SCHEMA, REVIEW_SCHEMA, Label, ReviewItem

STAGE_FILES = {"label": "label.md", "review": "review.md", "finalize": "finalize.md"}
SCHEMAS = {"label": LABEL_SCHEMA, "review": REVIEW_SCHEMA, "finalize": FINALIZE_SCHEMA}


@dataclass(frozen=True)
class Prompts:
    """The guideline and the three stage instructions, with a combined hash."""

    guideline: str
    stages: Mapping[str, str]

    @classmethod
    def load(cls, prompts_dir: str | Path) -> Prompts:
        """Read guideline.md and the stage files from ``prompts_dir``."""
        d = Path(prompts_dir)
        return cls(
            guideline=(d / "guideline.md").read_text(encoding="utf-8"),
            stages={k: (d / f).read_text(encoding="utf-8") for k, f in STAGE_FILES.items()},
        )

    @property
    def sha256(self) -> str:
        """Hash of every prompt file, recorded with each run."""
        h = hashlib.sha256(self.guideline.encode("utf-8"))
        for k in sorted(self.stages):
            h.update(k.encode("utf-8") + b"\0" + self.stages[k].encode("utf-8"))
        return h.hexdigest()


def _fmt_labels(labels: Sequence[Label]) -> str:
    return ", ".join(f"{lab.aspect.value}/{lab.polarity.value}" for lab in labels) or "[]"


def render_batch(
    batch: Batch,
    labels: Mapping[str, Sequence[Label]] | None = None,
    reviews: Mapping[str, Sequence[ReviewItem]] | None = None,
    only_ids: set[str] | None = None,
) -> str:
    """The user message for one batch.

    ``labels`` adds the stage-1 labels (stages 2 and 3); ``reviews`` adds every reviewer's
    verdict per sentence (stage 3); ``only_ids`` limits the listed sentences (stage 3 lists
    flagged sentences only, but still shows every review in full for context).
    """
    out = [f"<game>{escape(batch.game_name)}</game>", ""]
    for review in batch.reviews:
        listed = [s for s in review.sentences if only_ids is None or s.sentence_id in only_ids]
        if not listed:
            continue
        rec = "yes" if review.recommended else "no"
        out.append(f"<review id={quoteattr(str(review.review_id))} recommended={quoteattr(rec)}>")
        out.append(escape(review.text.strip()))
        out.append("</review>")
        out.append(f"<sentences review={quoteattr(str(review.review_id))}>")
        for s in listed:
            out.append(f"[{s.sentence_id}] {escape(s.sentence)}")
            if labels is not None:
                out.append(f"    labels: {_fmt_labels(labels[s.sentence_id])}")
            if reviews is not None:
                for n, item in enumerate(reviews.get(s.sentence_id, []), 1):
                    detail = (
                        f"{item.verdict.value}"
                        if item.verdict.value == "ok"
                        else f"flag ({item.issue.value}): {item.reasoning} "
                        f"-> suggested: {_fmt_labels(item.suggested_labels)}"
                    )
                    out.append(f"    reviewer {n}: {escape(detail)}")
        out.append("</sentences>")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def custom_id(stage: str, batch_id: str, reviewer: int | None = None) -> str:
    """Batch API custom_id (``[a-zA-Z0-9_-]{1,64}``), e.g. ``review2-289690-003``."""
    suffix = "" if reviewer is None else str(reviewer)
    return f"{stage}{suffix}-{batch_id}"


def build_params(stage: str, prompts: Prompts, user_text: str, cfg: Mapping[str, Any]) -> dict:
    """Messages API params for one request (``cfg`` is the stage section of labeling.yaml)."""
    return {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "system": [
            {"type": "text", "text": prompts.guideline},
            {
                "type": "text",
                "text": prompts.stages[stage],
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [{"role": "user", "content": user_text}],
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": cfg["effort"],
            "format": {"type": "json_schema", "schema": SCHEMAS[stage]},
        },
    }
