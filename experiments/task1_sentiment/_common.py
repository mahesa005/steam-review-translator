"""Helpers shared by the Task 1 data-prep scripts (not part of the package)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")


def clean_text(s: pd.Series) -> pd.Series:
    """Collapse whitespace and strip. NaN -> empty string."""
    return s.fillna("").astype(str).str.replace(_WS, " ", regex=True).str.strip()


def detect_language(texts: pd.Series, max_chars: int = 500) -> pd.Series:
    """ISO 639-1 code per text using py3langid (offline).

    Only the first ``max_chars`` characters are used: enough to tell the language, and it
    keeps long reviews from dominating the runtime.
    """
    import py3langid

    return pd.Series(
        [py3langid.classify(t[:max_chars])[0] if t else "" for t in texts.tolist()],
        index=texts.index,
    )


def cap_per_group(df: pd.DataFrame, key: str, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random sample of at most ``cap`` rows per ``key`` value (keeps all columns)."""
    shuffled = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    return shuffled.groupby(key, sort=False).head(cap)


def stratified_pick(
    games: pd.DataFrame, share_col: str, n: int, strata: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Pick ``n`` games spread evenly over ``strata`` quantile bins of ``share_col``.

    Bins that are too small give all their games; the shortfall is filled from the
    remaining games at random. Adds a ``stratum`` column (0 = lowest share).
    """
    if len(games) < n:
        raise ValueError(f"only {len(games)} eligible games, need {n}")
    games = games.copy()
    games["stratum"] = pd.qcut(games[share_col].rank(method="first"), strata, labels=False)
    per = n // strata
    picked: list[pd.DataFrame] = []
    for _, group in games.groupby("stratum"):
        k = min(per, len(group))
        picked.append(group.sample(n=k, random_state=int(rng.integers(1 << 31))))
    chosen = pd.concat(picked)
    if len(chosen) < n:
        rest = games.drop(chosen.index)
        extra = rest.sample(n=n - len(chosen), random_state=int(rng.integers(1 << 31)))
        chosen = pd.concat([chosen, extra])
    return chosen


def split_games(
    games: pd.DataFrame, fractions: dict[str, float], rng: np.random.Generator
) -> pd.Series:
    """Assign each game to a split, stratified by the ``stratum`` column."""
    names = list(fractions)
    assignment = pd.Series(index=games.index, dtype=object)
    for _, group in games.groupby("stratum"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        cuts = np.cumsum([round(fractions[n] * len(idx)) for n in names[:-1]])
        for name, part in zip(names, np.split(idx, cuts), strict=True):
            assignment.loc[part] = name
    return assignment


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
