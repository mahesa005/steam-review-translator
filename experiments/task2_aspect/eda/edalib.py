"""Shared helpers for the Task 2 EDA scripts. Research code, NOT part of the library.

Pure-Python parts (text filters, run-on chunking) have no heavy dependencies so they can be
unit-tested in CI; spaCy / fastText / polars are imported lazily inside the functions that
need them.

Sentence splitting and HTML decoding live in ``steam_review_translator.aspect.sentences``
(re-exported here). Candidate for promotion: ``LemmaLexiconMatcher``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import unicodedata
import zlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from steam_review_translator.aspect.lexicon import Lexicon, LexiconMatcher, load_lexicon, tokenize
from steam_review_translator.aspect.sentences import (  # noqa: F401 - re-exported for EDA scripts
    Sentence,
    chunk_runon,
    iter_sentences,
    normalize_ws,
    split_segments,
    unescape_html,
)
from steam_review_translator.core.config import get_path, load_yaml, project_root
from steam_review_translator.core.logging import get_logger
from steam_review_translator.core.schemas import Aspect

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.figure import Figure

EDA_DIR = Path(__file__).resolve().parent
CONFIG_PATH = "configs/task2/eda.yaml"

#: ``--smoke`` on any script: tiny input, ``*_smoke`` interim files, outputs to artifacts/.
SMOKE = "--smoke" in sys.argv
FORCE = "--force" in sys.argv
_SUFFIX = "_smoke" if SMOKE else ""
OUT_DIR = get_path("artifacts") / "eda" / "smoke_outputs" if SMOKE else EDA_DIR / "outputs"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"

#: The nine real aspects (``overall`` and ``other`` are non-aspect labels).
ASPECTS: tuple[str, ...] = tuple(a.value for a in Aspect if a not in (Aspect.OVERALL, Aspect.OTHER))
LABELS: tuple[str, ...] = (*ASPECTS, Aspect.OVERALL.value)


# --------------------------------------------------------------------------------------
# config / paths / io
# --------------------------------------------------------------------------------------
def load_cfg() -> dict[str, Any]:
    """Load ``configs/task2/eda.yaml``."""
    return load_yaml(CONFIG_PATH)


def logger(script: str) -> Any:
    """Logger named after the EDA script (uses the package logging setup)."""
    return get_logger(f"eda.{script}")


def interim(name: str) -> Path:
    """Path under ``data/interim/task2_eda/`` (created)."""
    d = get_path("interim") / "task2_eda"
    d.mkdir(parents=True, exist_ok=True)
    return d / _suffixed(name)


def processed(name: str) -> Path:
    """Path under ``data/processed/`` (created)."""
    d = get_path("processed")
    d.mkdir(parents=True, exist_ok=True)
    return d / _suffixed(name)


def cache(name: str) -> Path:
    """Path under ``artifacts/eda/`` for heavy caches (created)."""
    d = get_path("artifacts") / "eda"
    d.mkdir(parents=True, exist_ok=True)
    return d / _suffixed(name)


def _suffixed(name: str) -> str:
    p = Path(name)
    return f"{p.stem}{_SUFFIX}{p.suffix}"


def is_cached(path: Path, log: Any) -> bool:
    """True (and log it) if ``path`` exists and ``--force`` was not given."""
    if path.exists() and not FORCE:
        log.info("cached: %s (use --force to rebuild)", path.name)
        return True
    return False


def raw_file(cfg: dict[str, Any]) -> Path:
    """Absolute path of the raw dataset."""
    return project_root() / cfg["input"]["raw_file"]


def save_table(df: pl.DataFrame, name: str, log: Any = None) -> Path:
    """Write a committed output table to ``outputs/tables/<name>.csv``."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.write_csv(path, float_precision=4)
    if log is not None:
        log.info("table %s: %d rows -> %s", name, df.height, path)
    return path


def save_json(obj: Any, name: str) -> Path:
    """Write a small committed JSON summary to ``outputs/tables/<name>.json``."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.json"
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def save_fig(fig: Figure, name: str, log: Any = None) -> Path:
    """Save a figure to ``outputs/figures/<name>.png`` (small, 110 dpi) and close it."""
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    if log is not None:
        log.info("figure %s -> %s", name, path)
    return path


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def timed(log: Any, what: str) -> Iterator[None]:
    """Log start/end and wall time of a step."""
    start = time.perf_counter()
    log.info("%s ...", what)
    yield
    log.info("%s done in %.1f s", what, time.perf_counter() - start)


# --------------------------------------------------------------------------------------
# text filters (pure python, tested in tests/unit/test_eda_filters.py)
# --------------------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)
_REPEAT_TEMPLATE = r"(\S)\1{%d,}"
# Characters typical of ASCII/Unicode art: box drawing, blocks, braille, geometric shapes.
_ART_RANGES = ((0x2500, 0x259F), (0x25A0, 0x25FF), (0x2800, 0x28FF))
# Steam's profanity filter renders swear words as hearts; don't count those as art.
_IGNORED_SYMBOLS = {"♥"}


@dataclass(frozen=True)
class FilterParams:
    """Thresholds for :func:`text_flags` (see ``filters`` in eda.yaml)."""

    short_min_chars: int = 20
    short_min_words: int = 5
    repeated_char_run: int = 10
    min_alnum_ratio: float = 0.5
    ascii_art_min_chars: int = 40
    ascii_art_symbol_ratio: float = 0.4
    self_repeat_min_chars: int = 200
    self_repeat_max_ratio: float = 0.15

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> FilterParams:
        """Build from the ``filters`` section of eda.yaml (unknown keys ignored)."""
        f = cfg["filters"]
        return cls(**{k: f[k] for k in cls.__dataclass_fields__ if k in f})


def _is_art_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ART_RANGES)


def text_flags(text: str, p: FilterParams) -> dict[str, bool]:
    """Quality flags for one review.

    - ``empty``: nothing but whitespace.
    - ``short``: < ``short_min_chars`` chars or < ``short_min_words`` words.
    - ``repeated_chars``: one non-space character repeated ``repeated_char_run``+ times.
    - ``low_alnum``: letters+digits are < ``min_alnum_ratio`` of non-space characters.
    - ``ascii_art``: long text where art/box/braille or non-word symbols dominate.
    - ``self_repeat``: long text that compresses extremely well (a phrase pasted many times).

    Corpus-level copy-paste (the same long text posted many times) is computed separately
    because it needs global counts.
    """
    norm = normalize_ws(text)
    n_chars = len(norm)
    flags = {
        "empty": n_chars == 0,
        "short": n_chars < p.short_min_chars or len(_WORD.findall(norm)) < p.short_min_words,
        "repeated_chars": False,
        "low_alnum": False,
        "ascii_art": False,
        "self_repeat": False,
    }
    if n_chars == 0:
        return flags
    flags["repeated_chars"] = (
        re.search(_REPEAT_TEMPLATE % (p.repeated_char_run - 1), norm) is not None
    )
    visible = [c for c in norm if not c.isspace() and c not in _IGNORED_SYMBOLS]
    if visible:
        alnum_ratio = sum(c.isalnum() for c in visible) / len(visible)
        flags["low_alnum"] = alnum_ratio < p.min_alnum_ratio
        # symbols <= non-alnum share, so skip the per-char scan when art is impossible
        if n_chars >= p.ascii_art_min_chars and 1 - alnum_ratio >= p.ascii_art_symbol_ratio:
            art = sum(
                _is_art_char(c) or unicodedata.category(c)[0] in "SP" and c not in ".,!?'\""
                for c in visible
            )
            flags["ascii_art"] = art / len(visible) >= p.ascii_art_symbol_ratio
    if n_chars >= p.self_repeat_min_chars:
        raw = norm.encode("utf-8")
        flags["self_repeat"] = len(zlib.compress(raw, 6)) / len(raw) < p.self_repeat_max_ratio
    return flags


SPAM_FLAGS: tuple[str, ...] = (
    "repeated_chars",
    "low_alnum",
    "ascii_art",
    "self_repeat",
    "copy_paste",
)


# --------------------------------------------------------------------------------------
# language id (fastText lid.176)
# --------------------------------------------------------------------------------------
def LID_MODEL_PATH() -> Path:  # noqa: N802 - constant-like accessor
    """Where the fastText language-ID model is cached (shared by smoke and full runs)."""
    return get_path("artifacts") / "eda" / "lid.176.ftz"


class LangDetector:
    """fastText ``lid.176.ftz`` language identifier with batch prediction."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Download the model to ``artifacts/eda/`` once, then load it."""
        import urllib.request

        import fasttext

        path = LID_MODEL_PATH()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            urllib.request.urlretrieve(cfg["language"]["model_url"], path)
        self.model = fasttext.load_model(str(path))

    def predict(self, texts: Sequence[str]) -> tuple[list[str], list[float]]:
        """Return (language code, probability) per text. Empty text -> ("", 0.0)."""
        clean = [normalize_ws(t) for t in texts]
        idx = [i for i, t in enumerate(clean) if t]
        langs = [""] * len(clean)
        probs = [0.0] * len(clean)
        # fasttext-predict's list path returns labels without probabilities, so go per text
        for i in idx:
            labels, scores = self.model.predict(clean[i], k=1)
            if labels:
                langs[i] = labels[0].removeprefix("__label__")
                probs[i] = float(scores[0])
        return langs, probs


# --------------------------------------------------------------------------------------
# sentence splitting + lemmatization
# --------------------------------------------------------------------------------------
def load_nlp(model: str = "en_core_web_sm") -> Any:
    """spaCy pipeline: tokenizer + tagger + lemmatizer + rule-based sentencizer.

    Sentence boundaries are the library splitter's (``iter_sentences`` below is imported from
    ``steam_review_translator.aspect.sentences``); the model only adds lemmas.
    """
    import spacy

    nlp = spacy.load(model, exclude=["parser", "ner"])
    nlp.add_pipe("sentencizer")
    return nlp


# --------------------------------------------------------------------------------------
# lexicon matching on surface + lemma forms (wraps the library matcher)
# --------------------------------------------------------------------------------------
def load_lexicon_terms(path: str | Path) -> dict[str, list[str]]:
    """Aspect value -> normalized keywords, straight from the YAML lexicon."""
    return {a.value: list(ws) for a, ws in load_lexicon(path).terms.items()}


def load_lexicon_ambiguous(path: str | Path) -> frozenset[str]:
    """The lexicon's ``ambiguous`` words."""
    return load_lexicon(path).ambiguous


class LemmaLexiconMatcher:
    """Run the library's whole-token :class:`LexiconMatcher` on surface AND lemma forms.

    Lexicon terms are lemmatized with the same spaCy pipeline, so "crashes" matches "crash"
    and "characters" matches "character". Ambiguous words stay excluded in both forms.
    """

    def __init__(self, nlp: Any, lexicon_path: str | Path) -> None:
        """Build surface and lemma matchers from the YAML lexicon."""
        base = load_lexicon(lexicon_path)
        self.lexicon = base

        def lem(s: str) -> str:
            # normalize exactly like the library matcher tokenizes its input
            return " ".join(tokenize(" ".join(t.lemma_.lower() for t in nlp(s))))

        lemma_terms = {a: tuple(dict.fromkeys(lem(w) for w in ws)) for a, ws in base.terms.items()}
        lemma_amb = frozenset({*base.ambiguous, *(lem(w) for w in base.ambiguous)})
        self._surface = LexiconMatcher(base)
        self._lemma = LexiconMatcher(Lexicon(terms=lemma_terms, ambiguous=lemma_amb))
        self.term_of_lemma = {
            (a, lem(w)): w for a, ws in base.terms.items() for w in ws
        }  # map lemma hit back to the lexicon keyword

    def match_terms(self, text: str, lemmas: str) -> list[tuple[str, str]]:
        """Distinct ``(aspect, keyword)`` hits from surface and lemma forms."""
        hits: dict[tuple[str, str], None] = {}
        for a, term in self._surface.match_terms(text):
            hits[(a.value, term)] = None
        for a, lemma_term in self._lemma.match_terms(lemmas):
            hits[(a.value, self.term_of_lemma.get((a, lemma_term), lemma_term))] = None
        return list(hits)
