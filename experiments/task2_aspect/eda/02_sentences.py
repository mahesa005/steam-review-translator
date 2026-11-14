"""Step 2: sentence-level view on a per-game sample of clean English reviews.

Sampling (stated in the report): for every candidate game (>= min_english_reviews clean
English reviews) draw `candidates.reviews_per_game` clean reviews uniformly at random
(seed 42). Every game contributes the same number of reviews, so per-game numbers in steps
3 and 5 are comparable and big games do not dominate.

Splitter: 2+ spaces / newlines (former line breaks in this dump) -> spaCy rule-based
sentencizer (en_core_web_sm tokenizer, parser disabled) -> run-on chunking at , ; : and
conjunctions for "sentences" longer than `runon_max_tokens` tokens.
"""

from __future__ import annotations

import sys

import edalib as E
import matplotlib.pyplot as plt
import polars as pl

PCTS = [5, 25, 50, 75, 95, 99]


def sample_reviews(cfg: dict, log) -> pl.DataFrame:
    idx = pl.read_parquet(E.interim("review_index.parquet")).filter("clean")
    cands = pl.read_csv(E.TABLES_DIR / "candidate_games.csv")
    k = cfg["candidates"]["reviews_per_game"]
    picked = (
        idx.filter(pl.col("app_id").is_in(cands["app_id"].implode()))
        .with_columns(r=pl.int_range(pl.len()).shuffle(seed=cfg["seed"]).over("app_id"))
        .filter(pl.col("r") < k)
        .select("review_id", "app_id", "review_score")
    )
    texts = (
        pl.scan_parquet(E.interim("reviews.parquet"))
        .filter(pl.col("review_id").is_in(picked["review_id"].implode()))
        .select("review_id", "review_text")
        .collect()
    )
    out = picked.join(texts, on="review_id").sort("app_id", "review_id")
    log.info(
        "sampled %s reviews from %d games (<= %d each)",
        f"{out.height:,}",
        out["app_id"].n_unique(),
        k,
    )
    return out


def main() -> int:
    cfg = E.load_cfg()
    log = E.logger("02_sentences")
    sc = cfg["sentences"]
    out = E.interim("sentences.parquet")
    if not E.is_cached(out, log):
        reviews = sample_reviews(cfg, log)
        reviews.select("review_id", "app_id", "review_score").write_parquet(
            E.interim("sentence_sample_reviews.parquet")
        )
        nlp = E.load_nlp(sc["spacy_model"])
        rows: dict[str, list] = {
            k: [] for k in ("review_id", "sent_idx", "text", "lemmas", "n_words", "runon_split")
        }
        items = zip(reviews["review_text"].to_list(), reviews["review_id"].to_list(), strict=True)
        with E.timed(log, f"split + lemmatize {reviews.height:,} reviews"):
            for i, s in enumerate(
                E.iter_sentences(
                    nlp,
                    items,
                    split_on_multispace=sc["split_on_multispace"],
                    runon_max_tokens=sc["runon_max_tokens"],
                    batch_size=sc["batch_size"],
                    n_process=1 if E.SMOKE else sc["n_process"],
                )
            ):
                rows["review_id"].append(s.key)
                rows["sent_idx"].append(s.sent_idx)
                rows["text"].append(s.text)
                rows["lemmas"].append(s.lemmas)
                rows["n_words"].append(s.n_words)
                rows["runon_split"].append(s.runon_split)
                if i % 200_000 == 0:
                    log.info("  %s sentences", f"{i:,}")
        sents = pl.DataFrame(rows).join(reviews.select("review_id", "app_id"), on="review_id")
        sents.write_parquet(out)

    sents = pl.read_parquet(out)
    reviews = pl.read_parquet(E.interim("sentence_sample_reviews.parquet"))
    short_t = sc["short_sentence_tokens"]
    per_review = (
        sents.group_by("review_id")
        .agg(pl.len().alias("n_sent"))
        .join(reviews.select("review_id"), on="review_id", how="right")
        .with_columns(pl.col("n_sent").fill_null(0))
    )
    # reviews whose text has no . ! ? at all (handled by run-on chunking / segments)
    no_punct = (
        pl.scan_parquet(E.interim("reviews.parquet"))
        .filter(pl.col("review_id").is_in(reviews["review_id"].implode()))
        .select((~pl.col("review_text").str.contains(r"[.!?]")).mean())
        .collect()
        .item()
    )
    summary = [
        ("sampled reviews", reviews.height),
        ("sampled games", reviews["app_id"].n_unique()),
        ("sentences", sents.height),
        ("sentences per review (mean)", per_review["n_sent"].mean()),
        ("sentences per review (median)", per_review["n_sent"].median()),
        ("reviews with a single sentence (share)", (per_review["n_sent"] == 1).mean()),
        ("reviews without any . ! ? (share)", no_punct),
        ("sentences produced by run-on chunking (share)", sents["runon_split"].mean()),
        (f"sentences with < {short_t} word tokens (share)", (sents["n_words"] < short_t).mean()),
        ("words per sentence (mean)", sents["n_words"].mean()),
    ]
    E.save_table(
        pl.DataFrame(
            summary,
            schema=["metric", "value"],
            orient="row",
            schema_overrides={"value": pl.Float64},
        ),
        "sentence_summary",
        log,
    )
    pct_rows = []
    for name, s in (
        ("sentences_per_review", per_review["n_sent"]),
        ("words_per_sentence", sents["n_words"]),
    ):
        pct_rows.append(
            {
                "metric": name,
                "mean": s.mean(),
                **{f"p{p}": s.quantile(p / 100, "nearest") for p in PCTS},
            }
        )
    E.save_table(pl.DataFrame(pct_rows), "sentence_percentiles", log)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(per_review["n_sent"].clip(0, 40).to_numpy(), bins=40, color="#4c72b0")
    axes[0].set_xlabel("sentences per review (clipped 40)")
    axes[0].set_yscale("log")
    axes[1].hist(sents["n_words"].clip(0, 60).to_numpy(), bins=60, color="#55a868")
    axes[1].axvline(short_t, color="k", ls="--", lw=1)
    axes[1].set_xlabel("word tokens per sentence (clipped 60)")
    E.save_fig(fig, "sentence_distributions", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
