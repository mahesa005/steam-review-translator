"""Aspect lexicon (YAML) and a whole-token matcher.

Stub status: the matcher is a working baseline (exact whole-token / multi-word match, no
lemmatisation, no context). TODO(Mahesa): lemmatise, handle inflections ("crashed"), and
use context to resolve the ``ambiguous`` words instead of ignoring them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steam_review_translator.core.config import load_yaml
from steam_review_translator.core.schemas import Aspect

DEFAULT_LEXICON = "configs/task2/lexicon.yaml"

# Keeps "co-op", "10/10", "don't" as single tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into word tokens."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Lexicon:
    """Aspect -> trigger terms, plus words that must never trigger an aspect on their own."""

    terms: dict[Aspect, tuple[str, ...]]
    ambiguous: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Lexicon:
        """Build from ``{"aspects": {name: [terms]}, "ambiguous": [words]}``."""
        aspects = data.get("aspects") or {}
        if not isinstance(aspects, Mapping):
            raise ValueError("lexicon 'aspects' must be a mapping of aspect -> list of terms")
        terms: dict[Aspect, tuple[str, ...]] = {}
        for name, words in aspects.items():
            try:
                aspect = Aspect(str(name).strip().lower())
            except ValueError:
                valid = [a.value for a in Aspect]
                raise ValueError(f"unknown aspect {name!r} in lexicon; valid: {valid}") from None
            terms[aspect] = tuple(_norm(w) for w in (words or []) if _norm(w))
        ambiguous = frozenset(_norm(w) for w in (data.get("ambiguous") or []) if _norm(w))
        return cls(terms=terms, ambiguous=ambiguous)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Lexicon:
        """Load from a YAML file (see ``configs/task2/lexicon.yaml``)."""
        return cls.from_dict(load_yaml(path))


def load_lexicon(path: str | Path | None = None) -> Lexicon:
    """Load the lexicon, defaulting to ``configs/task2/lexicon.yaml``."""
    return Lexicon.from_yaml(path or DEFAULT_LEXICON)


def _norm(word: Any) -> str:
    return " ".join(tokenize(str(word)))


class LexiconMatcher:
    """Find aspect mentions by whole-token (and multi-word) lexicon lookup."""

    def __init__(self, lexicon: Lexicon) -> None:
        """Index every lexicon term by its token tuple, skipping ambiguous single words."""
        self._index: dict[tuple[str, ...], list[Aspect]] = {}
        for aspect, words in lexicon.terms.items():
            for word in words:
                key = tuple(word.split())
                if len(key) == 1 and key[0] in lexicon.ambiguous:
                    continue
                bucket = self._index.setdefault(key, [])
                if aspect not in bucket:
                    bucket.append(aspect)
        self._max_len = max((len(k) for k in self._index), default=0)

    def match_terms(self, sentence: str) -> list[tuple[Aspect, str]]:
        """Return every ``(aspect, matched term)`` in order of appearance."""
        tokens = tokenize(sentence)
        hits: list[tuple[Aspect, str]] = []
        for start in range(len(tokens)):
            for n in range(1, min(self._max_len, len(tokens) - start) + 1):
                key = tuple(tokens[start : start + n])
                for aspect in self._index.get(key, []):
                    hits.append((aspect, " ".join(key)))
        return hits

    def match(self, sentence: str) -> list[Aspect]:
        """Return the distinct aspects mentioned in ``sentence``, in order of first mention."""
        return list(dict.fromkeys(aspect for aspect, _ in self.match_terms(sentence)))
