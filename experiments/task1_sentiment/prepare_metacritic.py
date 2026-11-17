"""Build the Task 1 Metacritic dataset: label, select 200 games, split by game.

Input : data/raw/metacritic/games_reviews.csv (or the downloaded metacritic-games.zip)
        from Kaggle davutb/metacritic-games
Output: data/processed/task1/metacritic.parquet   one row per review
        data/processed/task1/metacritic_games.csv  one row per selected game
        data/processed/task1/metacritic_manifest.json  filters, counts, label shares

Steps (all parameters in configs/task1/data.yaml):
1. keep user reviews; clean whitespace; drop empty / < min_words / duplicate (game, text)
2. keep English (py3langid)
3. label from score: 0-4 negative, 5-7 neutral, 8-10 positive
4. eligible games: >= min_reviews usable reviews; pick n_games stratified by positive share
5. cap each game at max_reviews_per_game (random)
6. split games 70/15/15, stratified by the same positive-share bins
"""

from __future__ import annotations

import time
import zipfile

import numpy as np
import pandas as pd
from _common import (
    cap_per_group,
    clean_text,
    detect_language,
    split_games,
    stratified_pick,
    write_json,
)

from steam_review_translator.core.config import get_path, load_yaml
from steam_review_translator.sentiment.labels import score_to_polarity


def main() -> None:
    cfg_all = load_yaml("configs/task1/data.yaml")
    cfg = cfg_all["metacritic"]
    rng = np.random.default_rng(cfg_all["seed"])
    out_dir = get_path("processed") / cfg_all["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    t0 = time.time()

    raw_dir = get_path("raw") / cfg["raw_dir"]
    usecols = ["id", "title", "quote", "score", "platform", "review_type"]
    if (raw_dir / "games_reviews.csv").is_file():
        df = pd.read_csv(raw_dir / "games_reviews.csv", usecols=usecols)
    else:
        with zipfile.ZipFile(raw_dir / "metacritic-games.zip") as z:
            df = pd.read_csv(z.open("games_reviews.csv"), usecols=usecols)
    counts["raw_rows"] = len(df)
    df = df[df.review_type == cfg["review_type"]].drop(columns="review_type")
    counts["user_reviews"] = len(df)

    df["text"] = clean_text(df.pop("quote"))
    df = df[df.score.notna() & (df.text.str.split().str.len() >= cfg["min_words"])]
    counts["after_min_words"] = len(df)
    df = df.drop_duplicates(["id", "text"])
    counts["after_dedup"] = len(df)
    print(f"[{time.time() - t0:.0f}s] {len(df):,} reviews before language filter")

    # Language detection is the slow step: cache it in data/interim/ keyed by row index.
    lang_cache = get_path("interim") / cfg_all["output_subdir"] / "metacritic_lang.parquet"
    if lang_cache.is_file():
        df["lang"] = pd.read_parquet(lang_cache)["lang"].reindex(df.index)
    if "lang" not in df or df.lang.isna().any():
        df["lang"] = detect_language(df.text)
        lang_cache.parent.mkdir(parents=True, exist_ok=True)
        df[["lang"]].to_parquet(lang_cache)
    lang_share = df.lang.value_counts(normalize=True).head(10).round(4).to_dict()
    df = df[df.lang == cfg["language"]].drop(columns="lang")
    counts["after_language"] = len(df)
    print(f"[{time.time() - t0:.0f}s] {len(df):,} English reviews")

    df["score"] = df.score / 10.0  # stored as 0-100, user scores are 0-10
    df["label"] = [
        score_to_polarity(
            s, negative_max=cfg["negative_max"], positive_min=cfg["positive_min"]
        ).value
        for s in df.score
    ]

    games = (
        df.groupby("id")
        .agg(
            game=("title", "first"),
            n_reviews=("text", "size"),
            positive_share=("label", lambda s: (s == "positive").mean()),
            negative_share=("label", lambda s: (s == "negative").mean()),
        )
        .query("n_reviews >= @cfg['min_reviews']")
    )
    counts["eligible_games"] = len(games)
    games = stratified_pick(games, "positive_share", cfg["n_games"], cfg["strata"], rng)
    games["split"] = split_games(games, cfg["split"], rng)

    df = df[df.id.isin(games.index)]
    df = cap_per_group(df, "id", cfg["max_reviews_per_game"], rng)
    df = df.rename(columns={"id": "game_id", "title": "game"})
    df["split"] = df.game_id.map(games.split)
    df["review_id"] = "mc_" + df.index.astype(str)
    df = df[["review_id", "game_id", "game", "platform", "text", "score", "label", "split"]]
    df = df.sort_values(["split", "game_id", "review_id"]).reset_index(drop=True)
    counts["final_reviews"] = len(df)

    games["n_used"] = df.groupby("game_id").size()
    games = games.reset_index().rename(columns={"id": "game_id"})
    df.to_parquet(out_dir / "metacritic.parquet", index=False)
    games.to_csv(out_dir / "metacritic_games.csv", index=False)

    manifest = {
        "config": cfg,
        "seed": cfg_all["seed"],
        "counts": counts,
        "language_share_before_filter": lang_share,
        "games_per_split": games.split.value_counts().to_dict(),
        "reviews_per_split": df.split.value_counts().to_dict(),
        "label_share_overall": df.label.value_counts(normalize=True).round(4).to_dict(),
        "label_share_per_split": {
            s: g.label.value_counts(normalize=True).round(4).to_dict()
            for s, g in df.groupby("split")
        },
        "words_per_review": df.text.str.split()
        .str.len()
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .round(1)
        .to_dict(),
    }
    write_json(out_dir / "metacritic_manifest.json", manifest)
    print(f"[{time.time() - t0:.0f}s] wrote {len(df):,} reviews / {len(games)} games to {out_dir}")
    print(manifest["games_per_split"], manifest["reviews_per_split"])
    print("labels:", manifest["label_share_overall"])


if __name__ == "__main__":
    main()
