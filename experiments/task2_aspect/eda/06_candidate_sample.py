"""Step 6: reproducible candidate labeling sample (NOT labeled). Data -> data/processed/.

For each selected game (train/val/test, not backups): `labeling_sample.reviews_per_game`
clean English reviews drawn uniformly at random from the game's full clean pool (seed 42),
HTML entities decoded, split with the same splitter as step 2.
Ids: review_id = row number in the raw CSV; sentence_id = "<review_id>-<k>" (k = 0-based
sentence index within the review). Both are stable for a fixed config + splitter.
"""

from __future__ import annotations

import json
import sys

import edalib as E
import polars as pl


def main() -> int:
    cfg = E.load_cfg()
    log = E.logger("06_candidate_sample")
    sc = cfg["sentences"]
    k = cfg["labeling_sample"]["reviews_per_game"]
    sel = pl.read_csv(E.OUT_DIR / "selected_games.csv").filter(pl.col("split") != "backup")
    names = dict(zip(sel["game_id"], sel["name"], strict=True))
    splits = dict(zip(sel["game_id"], sel["split"], strict=True))

    idx = pl.read_parquet(E.interim("review_index.parquet")).filter(
        pl.col("clean") & pl.col("app_id").is_in(sel["game_id"].implode())
    )
    picked = (
        idx.with_columns(r=pl.int_range(pl.len()).shuffle(seed=cfg["seed"]).over("app_id"))
        .filter(pl.col("r") < k)
        .select("review_id", "app_id", "review_score")
    )
    texts = (
        pl.scan_parquet(E.interim("reviews.parquet"))
        .filter(pl.col("review_id").is_in(picked["review_id"].implode()))
        .select("review_id", "review_text")
        .collect()
    )
    reviews = picked.join(texts, on="review_id").sort("app_id", "review_id")
    log.info(
        "labeling sample: %d reviews from %d games", reviews.height, reviews["app_id"].n_unique()
    )

    nlp = E.load_nlp(sc["spacy_model"])
    meta = {
        r["review_id"]: r for r in reviews.select("review_id", "app_id", "review_score").to_dicts()
    }
    rows = []
    # The dump still contains HTML entities (&amp; &gt; &lt;); decode them so the labeling
    # sample shows the text a human (and an LLM) would read. Earlier steps keep the raw text.
    texts_clean = [E.unescape_html(t) for t in reviews["review_text"].to_list()]
    items = zip(texts_clean, reviews["review_id"].to_list(), strict=True)
    for s in E.iter_sentences(
        nlp,
        items,
        split_on_multispace=sc["split_on_multispace"],
        runon_max_tokens=sc["runon_max_tokens"],
        batch_size=sc["batch_size"],
    ):
        m = meta[s.key]
        rows.append(
            {
                "game_id": m["app_id"],
                "game_name": names[m["app_id"]],
                "split": splits[m["app_id"]],
                "review_id": s.key,
                "sentence_id": f"{s.key}-{s.sent_idx}",
                "sentence": s.text,
                "recommended": m["review_score"] == 1,
                "n_words": s.n_words,
            }
        )
    out = pl.DataFrame(rows).sort("game_id", "review_id")
    out = (
        out.with_columns(
            pl.col("sentence_id").str.split("-").list.last().cast(pl.Int32).alias("_k")
        )
        .sort("game_id", "review_id", "_k")
        .drop("_k")
    )
    csv_path, jsonl_path = (
        E.processed("candidate_sample.csv"),
        E.processed("candidate_sample.jsonl"),
    )
    out.write_csv(csv_path)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in out.iter_rows(named=True):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("wrote %d sentences -> %s, %s", out.height, csv_path, jsonl_path)

    short_t = sc["short_sentence_tokens"]
    lo, hi = cfg["labeling_sample"]["verify_seconds_per_sentence"]
    per_game = (
        out.group_by("game_id", "game_name", "split")
        .agg(
            reviews=pl.col("review_id").n_unique(),
            sentences=pl.len(),
            informative_sentences=(pl.col("n_words") >= short_t).sum(),
            recommended_share=pl.col("recommended").mean(),
        )
        .sort("split", "game_id", descending=[True, False])
    )
    total = per_game.select(
        pl.lit(None, pl.Int64).alias("game_id"),
        pl.lit("TOTAL").alias("game_name"),
        pl.lit("").alias("split"),
        pl.col("reviews").sum(),
        pl.col("sentences").sum(),
        pl.col("informative_sentences").sum(),
        pl.lit(None, pl.Float64).alias("recommended_share"),
    )
    table = pl.concat([per_game, total], how="vertical_relaxed").with_columns(
        verify_hours_low=pl.col("sentences") * lo / 3600,
        verify_hours_high=pl.col("sentences") * hi / 3600,
    )
    E.save_table(table, "candidate_sample_summary", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
