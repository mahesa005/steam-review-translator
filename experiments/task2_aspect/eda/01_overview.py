"""Step 1: corpus overview, language ID, quality flags, per-game stats.

Streams data/interim/task2_eda/reviews.parquet in batches and writes a text-free feature
table (review_features.parquet) that every later step joins on review_id.
No timestamp column exists in this dump, so there is no time analysis.
"""

from __future__ import annotations

import hashlib
import sys

import edalib as E
import matplotlib.pyplot as plt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

PCTS = [5, 25, 50, 75, 95, 99]


def text_hash(text: str) -> int:
    """Stable 63-bit hash of whitespace-normalized text (exact-duplicate detection)."""
    d = hashlib.blake2b(E.normalize_ws(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") >> 1


def build_features(cfg: dict, log) -> None:
    out = E.interim("review_features.parquet")
    if E.is_cached(out, log):
        return
    params = E.FilterParams.from_cfg(cfg)
    det = E.LangDetector(cfg)
    src = pq.ParquetFile(E.interim("reviews.parquet"))
    writer = None
    done = 0
    flag_names = ["empty", "short", "repeated_chars", "low_alnum", "ascii_art", "self_repeat"]
    with E.timed(log, "features (flags + language + hash)"):
        for batch in src.iter_batches(
            batch_size=200_000,
            columns=["review_id", "app_id", "review_text", "review_score", "review_votes"],
        ):
            texts = batch.column("review_text").to_pylist()
            norm = [E.normalize_ws(t) for t in texts]
            flags = [E.text_flags(t, params) for t in texts]
            langs, probs = det.predict(norm)
            table = pa.table(
                {
                    "review_id": batch.column("review_id"),
                    "app_id": batch.column("app_id"),
                    "review_score": batch.column("review_score"),
                    "review_votes": batch.column("review_votes"),
                    "n_chars": pa.array([len(t) for t in norm], pa.int32()),
                    "n_words": pa.array([len(t.split()) for t in norm], pa.int32()),
                    **{f: pa.array([fl[f] for fl in flags]) for f in flag_names},
                    "lang": pa.array(langs),
                    "lang_prob": pa.array(probs, pa.float32()),
                    "text_hash": pa.array([text_hash(t) for t in norm], pa.int64()),
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
            done += batch.num_rows
            log.info("  features: %s rows", f"{done:,}")
    if writer is not None:
        writer.close()


def main() -> int:
    cfg = E.load_cfg()
    log = E.logger("01_overview")
    build_features(cfg, log)
    f = cfg["filters"]
    min_prob = cfg["language"]["min_prob"]

    df = pl.read_parquet(E.interim("review_features.parquet"))
    n = df.height
    log.info("loaded features: %s rows", f"{n:,}")

    # corpus-level duplicates -> copy_paste flag
    dup_counts = df.group_by("text_hash").agg(
        pl.len().alias("dup_count"), pl.col("app_id").n_unique().alias("dup_games")
    )
    df = df.join(dup_counts, on="text_hash", how="left").with_columns(
        copy_paste=(pl.col("dup_count") >= f["copy_paste_min_count"])
        & (pl.col("n_chars") >= f["copy_paste_min_chars"]),
        is_english=(pl.col("lang") == "en") & (pl.col("lang_prob") >= min_prob),
    )
    df = df.with_columns(spam=pl.any_horizontal(*E.SPAM_FLAGS)).with_columns(
        clean=pl.col("is_english") & ~pl.col("short") & ~pl.col("spam") & ~pl.col("empty")
    )
    df.select(
        "review_id",
        "app_id",
        "review_score",
        "review_votes",
        "n_chars",
        "n_words",
        "is_english",
        "short",
        "spam",
        "clean",
        "dup_count",
    ).write_parquet(E.interim("review_index.parquet"))

    # ---------------- overview counts
    eng = df.filter("is_english")
    n_dup_rows = df.filter(pl.col("dup_count") > 1).height
    overview = [
        ("rows", n),
        ("columns", len(cfg["input"]["columns"])),
        ("games (distinct app_id)", df["app_id"].n_unique()),
        ("empty review_text", int(df["empty"].sum())),
        ("rows whose exact text appears more than once", n_dup_rows),
        ("distinct texts that are duplicated", dup_counts.filter(pl.col("dup_count") > 1).height),
        (f"english rows (fastText lid.176, p>={min_prob:.2f})", eng.height),
        ("english share", eng.height / n),
        ("english share among non-short reviews", df.filter(~pl.col("short"))["is_english"].mean()),
        ("clean english rows (english, not short, not spam)", int(df["clean"].sum())),
        ("clean share of all rows", df["clean"].mean()),
        ("timestamp column available", 0),
    ]
    E.save_table(
        pl.DataFrame(
            overview,
            schema=["metric", "value"],
            orient="row",
            schema_overrides={"value": pl.Float64},
        ),
        "overview_counts",
        log,
    )
    E.save_table(
        pl.DataFrame(
            {
                "column": cfg["input"]["columns"],
                "dtype": ["int32", "string", "int8", "int8"],
                "n_missing_or_empty": [0, int(df["empty"].sum()), 0, 0],
            }
        ),
        "missing_values",
        log,
    )

    # duplicates (short snippets only)
    top_dups = dup_counts.sort("dup_count", descending=True).head(25)
    reviews = pl.scan_parquet(E.interim("reviews.parquet"))
    snippet_ids = (
        df.filter(pl.col("text_hash").is_in(top_dups["text_hash"].implode()))
        .group_by("text_hash")
        .agg(pl.col("review_id").min())
    )
    snippets = (
        reviews.filter(pl.col("review_id").is_in(snippet_ids["review_id"].implode()))
        .select(
            "review_id", pl.col("review_text").str.strip_chars().str.slice(0, 80).alias("snippet")
        )
        .collect()
        .join(snippet_ids, on="review_id")
        .join(top_dups, on="text_hash")
        .sort("dup_count", descending=True)
        .select("snippet", "dup_count", "dup_games")
    )
    E.save_table(snippets, "duplicates_top", log)

    # ---------------- language
    lang = (
        df.group_by("lang")
        .agg(pl.len().alias("n"))
        .with_columns(share=pl.col("n") / n)
        .sort("n", descending=True)
    )
    E.save_table(lang.head(20), "language_share", log)

    # ---------------- lengths
    rows = []
    for pop_name, pop in (("all", df), ("english", eng), ("clean_english", df.filter("clean"))):
        for col in ("n_chars", "n_words"):
            s = pop[col]
            rows.append(
                {
                    "population": pop_name,
                    "metric": col,
                    "mean": s.mean(),
                    **{f"p{p}": s.quantile(p / 100, "nearest") for p in PCTS},
                }
            )
    E.save_table(pl.DataFrame(rows), "length_percentiles", log)

    flags_tbl = pl.DataFrame(
        [
            {
                "flag": fl,
                "n_all": int(df[fl].sum()),
                "share_all": df[fl].mean(),
                "n_english": int(eng[fl].sum()),
                "share_english": eng[fl].mean(),
            }
            for fl in ("empty", "short", *E.SPAM_FLAGS, "spam")
        ]
    )
    E.save_table(flags_tbl, "quality_flags", log)

    # ---------------- recommend flag / helpful votes
    score = (
        df.group_by("review_score", "review_votes")
        .agg(pl.len().alias("n"))
        .with_columns(share=pl.col("n") / n)
        .sort("review_score", "review_votes")
    )
    E.save_table(score, "score_distribution", log)

    # ---------------- per game
    games = (
        df.group_by("app_id")
        .agg(
            n_reviews=pl.len(),
            n_english=pl.col("is_english").sum(),
            n_clean=pl.col("clean").sum(),
            english_share=pl.col("is_english").mean(),
            short_share_en=pl.col("short").filter("is_english").mean(),
            spam_share_en=pl.col("spam").filter("is_english").mean(),
            pos_share_clean=(pl.col("review_score") == 1).filter("clean").mean(),
            helpful_share_clean=(pl.col("review_votes") == 1).filter("clean").mean(),
            median_chars_clean=pl.col("n_chars").filter("clean").median(),
            median_words_clean=pl.col("n_words").filter("clean").median(),
        )
        .sort("n_reviews", descending=True)
    )
    # committed table keeps games with >= 200 reviews (size limit); the all-games
    # distribution is in reviews_per_game_describe.csv
    E.save_table(games.filter(pl.col("n_reviews") >= 200), "game_stats", log)
    cands = games.filter(pl.col("n_clean") >= cfg["candidates"]["min_english_reviews"])
    E.save_table(cands.select("app_id", "n_clean"), "candidate_games", log)

    desc = []
    for col in ("n_reviews", "n_clean"):
        s = games[col]
        desc.append(
            {
                "metric": col,
                "games": s.len(),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                **{f"p{p}": s.quantile(p / 100, "nearest") for p in PCTS},
                "max": s.max(),
                "games_ge_200": int((s >= 200).sum()),
                "games_ge_1000": int((s >= 1000).sum()),
            }
        )
    E.save_table(pl.DataFrame(desc), "reviews_per_game_describe", log)
    E.save_table(
        games.head(30).select("app_id", "n_reviews", "n_english", "n_clean", "pos_share_clean"),
        "top30_games",
        log,
    )
    share_top = games.head(30)["n_reviews"].sum() / n
    E.save_json(
        {
            "share_of_reviews_in_top30_games": share_top,
            "share_of_reviews_in_top1pct_games": games.head(max(1, games.height // 100))[
                "n_reviews"
            ].sum()
            / n,
        },
        "concentration",
    )

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        games["n_reviews"].to_numpy(),
        bins=[10**x for x in [i / 4 for i in range(0, 25)]],
        color="#4c72b0",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reviews per game (log)")
    ax.set_ylabel("games (log)")
    ax.set_title(f"Reviews per game ({games.height:,} games)")
    E.save_fig(fig, "reviews_per_game_hist", log)

    top = games.head(30)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([str(a) for a in top["app_id"]][::-1], top["n_reviews"].to_numpy()[::-1], label="all")
    ax.barh(
        [str(a) for a in top["app_id"]][::-1],
        top["n_clean"].to_numpy()[::-1],
        label="clean English",
    )
    ax.set_xlabel("reviews")
    ax.set_ylabel("app_id")
    ax.legend()
    ax.set_title("Top 30 games")
    E.save_fig(fig, "top30_games", log)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(eng["n_words"].clip(0, 400).to_numpy(), bins=80, color="#55a868")
    ax.set_yscale("log")
    ax.set_xlabel("words per English review (clipped at 400)")
    ax.set_ylabel("reviews (log)")
    E.save_fig(fig, "review_length_hist", log)

    lt = lang.head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([x or "(empty)" for x in lt["lang"]], lt["share"].to_numpy(), color="#8172b2")
    ax.set_ylabel("share of all reviews")
    ax.set_title("Detected language (top 10)")
    E.save_fig(fig, "language_share", log)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cands["pos_share_clean"].to_numpy(), bins=40, color="#c44e52")
    ax.set_xlabel("share of recommended reviews (clean English)")
    ax.set_ylabel("candidate games")
    ax.set_title(f"Recommend share, {cands.height} candidate games")
    E.save_fig(fig, "recommend_share_per_game", log)
    log.info(
        "candidates (>= %d clean English reviews): %d games",
        cfg["candidates"]["min_english_reviews"],
        cands.height,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
