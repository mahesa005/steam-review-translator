"""Review preprocessing and sentence splitting (the one splitter used everywhere).

The pipeline, the Task 2 EDA and the labeling sample all split reviews with this module, so a
model trained on labeled sentences sees exactly the same sentences at inference time.

Steps for one review:
1. ``unescape_html``: decode HTML entities left in the Steam dump (``&amp;`` -> ``&``).
2. ``split_segments``: cut at line breaks and at 2+ spaces (the dump collapsed line breaks
   into runs of spaces).
3. spaCy rule-based sentencizer on each segment (``spacy.blank("en")``; no model download).
4. ``chunk_runon``: re-split "sentences" longer than ``runon_max_tokens`` tokens (reviews
   without punctuation) at ``, ; :`` and conjunctions.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

#: Default for step 4 (also ``sentences.runon_max_tokens`` in configs/task2/eda.yaml).
RUNON_MAX_TOKENS = 50

_WS = re.compile(r"\s+")
_MULTISPACE = re.compile(r"\s{2,}|\n+")
_CONJ_BREAKS = {"but", "and", "so", "because", "although", "though", "however", "also"}
_PUNCT_BREAKS = {",", ";", ":", "-", "--", "|", "/"}


def unescape_html(text: str, max_rounds: int = 3) -> str:
    """Decode HTML entities, repeating for double-encoded text (``&amp;amp;trade;`` -> ™)."""
    for _ in range(max_rounds):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to one space and strip."""
    return _WS.sub(" ", text).strip()


def split_segments(text: str, split_on_multispace: bool = True) -> list[str]:
    """Cut a review at line breaks (and, for this dump, at 2+ spaces = former line breaks)."""
    parts = _MULTISPACE.split(text) if split_on_multispace else [text]
    return [s for s in (normalize_ws(p) for p in parts) if s]


def chunk_runon(words: Sequence[str], max_tokens: int, min_chunk: int = 8) -> list[tuple[int, int]]:
    """Split a long token run (no sentence punctuation) into ``(start, end)`` ranges.

    Cuts before conjunctions and after ``, ; :`` once the current chunk has ``min_chunk``
    tokens; any chunk still > ``max_tokens`` is cut hard every ``max_tokens`` tokens.
    """
    n = len(words)
    if n <= max_tokens:
        return [(0, n)]
    spans: list[tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        lw = w.lower()
        if i - start >= min_chunk and lw in _CONJ_BREAKS:
            spans.append((start, i))
            start = i
        elif i + 1 - start >= min_chunk and lw in _PUNCT_BREAKS:
            spans.append((start, i + 1))
            start = i + 1
    if start < n:
        spans.append((start, n))
    out: list[tuple[int, int]] = []
    for s, e in spans:
        while e - s > max_tokens:
            out.append((s, s + max_tokens))
            s += max_tokens
        if e > s:
            out.append((s, e))
    return out


@lru_cache(maxsize=1)
def blank_nlp() -> Any:
    """Tokenizer + rule-based sentencizer only (``spacy.blank("en")``), built once."""
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


@dataclass(frozen=True)
class Sentence:
    """One sentence produced by :func:`iter_sentences`."""

    key: Any
    sent_idx: int
    text: str
    lemmas: str  # lowercase lemmas joined by single spaces (empty if nlp has no lemmatizer)
    n_words: int  # tokens that are not punctuation / whitespace
    runon_split: bool


def iter_sentences(
    nlp: Any,
    items: Iterable[tuple[str, Any]],
    *,
    split_on_multispace: bool = True,
    runon_max_tokens: int = RUNON_MAX_TOKENS,
    batch_size: int = 512,
    n_process: int = 1,
) -> Iterator[Sentence]:
    """Split ``(text, key)`` items into sentences, in input order per key (steps 2-4).

    ``text`` is used as given; decode HTML first with :func:`unescape_html` if needed. Any
    spaCy pipeline with a sentencizer works; boundaries depend only on the tokenizer and the
    sentencizer, so :func:`blank_nlp` and ``en_core_web_sm`` give identical sentences.
    """

    def segments() -> Iterator[tuple[str, Any]]:
        for text, key in items:
            for seg in split_segments(text, split_on_multispace):
                yield seg, key

    counters: dict[Any, int] = {}
    docs = nlp.pipe(segments(), as_tuples=True, batch_size=batch_size, n_process=n_process)
    for doc, key in docs:
        for sent in doc.sents:
            toks = [t for t in sent if not t.is_space]
            if not toks:
                continue
            ranges = chunk_runon([t.text for t in toks], runon_max_tokens)
            for s, e in ranges:
                chunk = toks[s:e]
                text = "".join(t.text_with_ws for t in chunk).strip()
                if not text:
                    continue
                idx = counters.get(key, 0)
                counters[key] = idx + 1
                yield Sentence(
                    key=key,
                    sent_idx=idx,
                    text=text,
                    lemmas=" ".join(t.lemma_.lower() for t in chunk),
                    n_words=sum(not t.is_punct for t in chunk),
                    runon_split=len(ranges) > 1,
                )


def split_reviews(texts: Sequence[str]) -> list[list[str]]:
    """Preprocess and split many reviews (steps 1-4); one list of sentences per review."""
    out: list[list[str]] = [[] for _ in texts]
    items = ((unescape_html(t), i) for i, t in enumerate(texts))
    for s in iter_sentences(blank_nlp(), items):
        out[s.key].append(s.text)
    return out


def split_sentences(text: str) -> list[str]:
    """Preprocess and split one review (steps 1-4)."""
    return split_reviews([text])[0]
