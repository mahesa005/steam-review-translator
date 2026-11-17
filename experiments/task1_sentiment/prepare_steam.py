"""Build the Task 1 Steam test set: 50 games x 200 reviews with recommend labels.

Input : data/raw/steam/steam.csv.bz2  (Zenodo record 1000885; no header:
        app_id, review_text, review_score (1 / -1), review_votes)
Output: data/processed/task1/steam.parquet          one row per review
        data/processed/task1/steam_games.csv        one row per selected app
        data/processed/task1/steam_manual_check.csv reviews for the label-noise check
        data/processed/task1/steam_manifest.json

Steps (parameters in configs/task1/data.yaml):
1. pass 1 over the dump: per-app counts and recommend share (after cheap filters)
2. pick n_games apps with >= min_reviews, stratified by recommend share
   (+ include_app_ids, e.g. the Task 2 games)
3. pass 2: collect those apps' reviews, keep English, dedupe, sample reviews_per_game
4. gold distribution per game = recommend share in the sampled reviews
5. export a random sample for manual label-noise annotation
6. (optional) fetch game names from the Steam store API (--no-names to skip)
"""

from __future__ import annotations

import argparse
import bz2
import json
import time
import urllib.request
from collections.abc import Iterator

import numpy as np
import pandas as pd
from _common import cap_per_group, clean_text, detect_language, stratified_pick, write_json

from steam_review_translator.core.config import get_path, load_yaml

COLUMNS = ["app_id", "text", "score", "votes"]
PLACEHOLDERS = {"Early Access Review"}
CHUNK = 500_000


def read_chunks(path: str) -> Iterator[pd.DataFrame]:
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        yield from pd.read_csv(
            fh, header=None, names=COLUMNS, usecols=["app_id", "text", "score"], chunksize=CHUNK
        )


def basic_filter(df: pd.DataFrame, min_words: int) -> pd.DataFrame:
    df = df[df.score.isin([1, -1])].copy()
    df["text"] = clean_text(df.text)
    df = df[~df.text.isin(PLACEHOLDERS)]
    return df[df.text.str.split().str.len() >= min_words]


def fetch_name(app_id: int) -> str | None:
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&filters=basic"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - fixed host
            data = json.load(resp).get(str(app_id), {})
        return data.get("data", {}).get("name") if data.get("success") else None
    except Exception:  # noqa: BLE001 - names are a convenience only
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-names", action="store_true", help="skip Steam store name lookup")
    args = parser.parse_args()

    cfg_all = load_yaml("configs/task1/data.yaml")
    cfg = cfg_all["steam"]
    rng = np.random.default_rng(cfg_all["seed"])
    out_dir = get_path("processed") / cfg_all["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = str(get_path("raw") / cfg["raw_file"])
    t0 = time.time()

    # Pass 1: per-app stats.
    stats: list[pd.DataFrame] = []
    raw_rows = 0
    for chunk in read_chunks(raw):
        raw_rows += len(chunk)
        chunk = basic_filter(chunk, cfg["min_words"])
        stats.append(
            chunk.assign(pos=chunk.score == 1)
            .groupby("app_id")
            .agg(n=("pos", "size"), n_pos=("pos", "sum"))
        )
        print(f"[{time.time() - t0:.0f}s] pass 1: {raw_rows:,} rows", flush=True)
    apps = pd.concat(stats).groupby(level=0).sum()
    apps["recommend_share"] = apps.n_pos / apps.n
    eligible = apps[apps.n >= cfg["min_reviews"]]
    print(f"{len(apps):,} apps, {len(eligible):,} with >= {cfg['min_reviews']} reviews")

    forced = [a for a in cfg.get("include_app_ids") or [] if a in apps.index]
    pool = eligible.drop(index=[a for a in forced if a in eligible.index])
    chosen = stratified_pick(
        pool, "recommend_share", cfg["n_games"] - len(forced), cfg["strata"], rng
    )
    if forced:
        chosen = pd.concat([apps.loc[forced].assign(stratum=-1), chosen])
    chosen_ids = set(chosen.index)

    # Pass 2: collect the chosen apps' reviews.
    parts: list[pd.DataFrame] = []
    for chunk in read_chunks(raw):
        chunk = chunk[chunk.app_id.isin(chosen_ids)]
        if len(chunk):
            parts.append(basic_filter(chunk, cfg["min_words"]))
    df = pd.concat(parts)
    df["row"] = df.index
    df = df.drop_duplicates(["app_id", "text"])
    # Over-sample before the language filter so each app still has enough reviews.
    df = cap_per_group(df, "app_id", cfg["reviews_per_game"] * 3, rng)
    df["lang"] = detect_language(df.text)
    df = df[df.lang == cfg["language"]].drop(columns="lang")
    df = cap_per_group(df, "app_id", cfg["reviews_per_game"], rng)
    print(f"[{time.time() - t0:.0f}s] pass 2: {len(df):,} reviews")

    df["recommended"] = df.score == 1
    df["label"] = np.where(df.recommended, "positive", "negative")
    df["review_id"] = "st_" + df.row.astype(str)
    df = df[["review_id", "app_id", "text", "recommended", "label"]]
    df = df.sort_values(["app_id", "review_id"]).reset_index(drop=True)

    games = chosen.rename(columns={"n": "n_dump", "recommend_share": "recommend_share_dump"})
    games = games.drop(columns="n_pos")
    games["n_used"] = df.groupby("app_id").size()
    games["gold_positive"] = df.groupby("app_id").recommended.mean()
    games.index.name = "app_id"
    games = games.reset_index()
    if not args.no_names:
        games["name"] = [fetch_name(int(a)) for a in games.app_id]
    short = games[games.n_used < cfg["reviews_per_game"]]
    if len(short):
        print(f"warning: {len(short)} apps have < {cfg['reviews_per_game']} usable reviews")

    manual = df.sample(n=min(cfg["manual_check_size"], len(df)), random_state=cfg_all["seed"])
    manual = manual.assign(manual_label="", note="")[
        ["review_id", "app_id", "recommended", "text", "manual_label", "note"]
    ]

    df.to_parquet(out_dir / "steam.parquet", index=False)
    games.to_csv(out_dir / "steam_games.csv", index=False)
    manual.to_csv(out_dir / "steam_manual_check.csv", index=False)
    write_json(
        out_dir / "steam_manifest.json",
        {
            "config": cfg,
            "seed": cfg_all["seed"],
            "raw_rows": raw_rows,
            "apps_in_dump": len(apps),
            "eligible_apps": len(eligible),
            "n_games": len(games),
            "n_reviews": len(df),
            "recommend_share_overall": round(float(df.recommended.mean()), 4),
            "gold_positive_per_game": games.gold_positive.describe().round(4).to_dict(),
            "words_per_review": df.text.str.split()
            .str.len()
            .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
            .round(1)
            .to_dict(),
        },
    )
    print(f"[{time.time() - t0:.0f}s] wrote {len(df):,} reviews / {len(games)} apps to {out_dir}")


if __name__ == "__main__":
    main()
