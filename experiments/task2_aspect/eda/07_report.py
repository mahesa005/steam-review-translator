"""Step 7: assemble outputs/eda_report.md from outputs/decisions.md + the CSVs in tables/.

Every table below is rendered straight from a CSV (source named under it), so every number
in the report is traceable. decisions.md is written by hand after reading these results.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys

import edalib as E
import polars as pl


def fmt(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if v == int(v) and abs(v) >= 10:
            return f"{int(v):,}"
        return f"{v:.3f}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v).replace("|", "/").replace("\n", " ")


def table(
    name: str,
    cols: list[str] | None = None,
    n: int | None = None,
    where: pl.Expr | None = None,
    sort: tuple[str, bool] | None = None,
) -> str:
    path = E.TABLES_DIR / f"{name}.csv"
    if not path.is_file():
        return f"_missing: tables/{name}.csv_\n"
    df = pl.read_csv(path, infer_schema_length=None)
    if where is not None:
        df = df.filter(where)
    if sort:
        df = df.sort(sort[0], descending=sort[1], nulls_last=True)
    if cols:
        df = df.select([c for c in cols if c in df.columns])
    total = df.height
    if n is not None:
        df = df.head(n)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(fmt(v) for v in row) + " |\n" for row in df.iter_rows())
    more = f" (first {df.height} of {total} rows)" if n is not None and total > df.height else ""
    return head + body + f"\n_Source: `tables/{name}.csv`{more}_\n"


def js(name: str) -> dict:
    p = E.TABLES_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def fig(name: str, caption: str) -> str:
    return f"![{caption}](figures/{name}.png)\n\n_{caption}_\n"


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        env["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["gpu"] = "none"
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.cuda.is_available()
    except ImportError:
        env["torch"] = "not installed"
    for mod in (
        "bertopic",
        "sentence_transformers",
        "spacy",
        "polars",
        "gensim",
        "hdbscan",
        "umap",
    ):
        try:
            env[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            env[mod] = "?"
    return env


def main() -> int:
    cfg = E.load_cfg()
    log = E.logger("07_report")
    env = environment()
    E.save_json(env, "environment")
    raw = js("raw_file_fingerprint")
    lex = js("lexicon_version")
    conc = js("concentration")
    split = js("split_objective")
    decisions = E.OUT_DIR / "decisions.md"
    dec = (
        decisions.read_text(encoding="utf-8") if decisions.is_file() else "_decisions.md missing_\n"
    )
    tc = cfg["topics"]
    main_cfg = f"mcs{tc['hdbscan_min_cluster_sizes'][0]}_reduce-none"
    st = cfg["sentences"]["short_sentence_tokens"]

    md: list[str] = []
    a = md.append
    a("# Task 2 EDA: Steam reviews for aspect-based opinion mining\n")
    a(
        f"Generated {dt.date.today().isoformat()} by `experiments/task2_aspect/eda/07_report.py`. "
        "Rerun: see `experiments/task2_aspect/eda/README.md`.\n"
    )
    a(
        f"- Raw data: `{raw.get('file')}`, {raw.get('bytes', 0):,} bytes, "
        f"sha256 `{raw.get('sha256')}`."
    )
    a(
        f"- Lexicon: `{lex.get('path')}` sha256 `{lex.get('sha256', '')[:16]}…` (git blob "
        f"`{lex.get('git_blob', '')[:10]}`). **Status: {lex.get('status')}. Every lexicon-based "
        "number (hit rates, aspect shares, purity, game scores, the selection itself) is "
        "PRELIMINARY.**"
    )
    a(
        f"- Environment: {env.get('platform')}, {env.get('cpu_count')} CPU threads, GPU "
        f"{env.get('gpu')}; torch {env.get('torch')} (CUDA used: {env.get('torch_cuda')}). "
        f"Config: `configs/task2/eda.yaml`, seed {cfg['seed']}. Full versions: "
        "`tables/environment.json`.\n"
    )
    a("## Decisions\n")
    a(dec.rstrip() + "\n")

    a("## 1. Corpus overview\n")
    a(
        "Schema (no header in the CSV; mapping confirmed by the Task 2 owner): `app_id`, "
        "`review_text`, `review_score` (1 recommended / -1 not), `review_votes` (1 = voted "
        "helpful at least once; metadata only). **No timestamp and no game-name column**, so "
        "there is no time-range analysis; names/genres come from the Steam store API for the "
        "shortlisted games only.\n"
    )
    a(table("overview_counts"))
    a(table("missing_values"))
    a("### Reviews per game\n")
    a(table("reviews_per_game_describe"))
    a(
        "Concentration: the top 30 games hold "
        f"{conc.get('share_of_reviews_in_top30_games', 0):.1%} "
        f"of all reviews; the top 1% of games hold "
        f"{conc.get('share_of_reviews_in_top1pct_games', 0):.1%} (`tables/concentration.json`).\n"
    )
    a(fig("reviews_per_game_hist", "Reviews per game (log-log)"))
    a(table("top30_games"))
    a(fig("top30_games", "Top 30 games by review count"))
    a("### Language\n")
    a(
        "Detector: fastText `lid.176.ftz` (top-1 label, English if `en` with p >= "
        f"{cfg['language']['min_prob']}). Very short texts are unreliable for any detector, so "
        "the share among non-short reviews is also reported in the overview table.\n"
    )
    a(table("language_share", n=12))
    a(fig("language_share", "Detected language"))
    a("### Length and quality\n")
    a(table("length_percentiles"))
    a(fig("review_length_hist", "Words per English review"))
    a(
        "Flags (definitions in `configs/task2/eda.yaml` → `filters` and `edalib.text_flags`): "
        "short = < 20 chars or < 5 words; spam = any of repeated_chars, low_alnum, ascii_art, "
        "self_repeat (compresses to < 15%), copy_paste (identical >= 40-char text posted >= 5 "
        "times). A review is **clean** if English, not short, not spam.\n"
    )
    a(table("quality_flags"))
    a("Most frequent exact-duplicate texts (snippets):\n")
    a(table("duplicates_top", n=15))
    a("### Recommend flag\n")
    a(table("score_distribution"))
    a(fig("recommend_share_per_game", "Share of recommended reviews per candidate game"))

    a("## 2. Sentence-level view\n")
    a(
        f"Sample: {cfg['candidates']['reviews_per_game']} clean English reviews per candidate game "
        f"(games with >= {cfg['candidates']['min_english_reviews']} clean English reviews), "
        "uniform "
        f"random, seed {cfg['seed']}. Splitter: line-break/2+-space segments → spaCy "
        f"`{cfg['sentences']['spacy_model']}` rule-based sentencizer → run-on chunking for "
        f"segments > {cfg['sentences']['runon_max_tokens']} tokens (at , ; : and conjunctions). "
        "Lemmas from the same spaCy pipeline.\n"
    )
    a(table("sentence_summary"))
    a(table("sentence_percentiles"))
    a(fig("sentence_distributions", "Sentences per review and words per sentence"))

    a("## 3. Lexicon hit rate (system A feasibility) — PRELIMINARY\n")
    a(
        "Matching: whole-token match of lexicon keywords/phrases on lowercase surface tokens "
        "and on spaCy lemmas (union), using the library `LexiconMatcher`. Ambiguous words "
        "(hard, easy, smooth, free, beautiful) never match on their own.\n"
    )
    a(table("lexicon_hit_rate"))
    a(table("lexicon_hits_per_aspect"))
    a(fig("lexicon_hits_per_aspect", "Hits per aspect"))
    a(table("aspects_per_sentence"))
    a("Keyword frequency (top 25 and keywords that never matched):\n")
    a(table("lexicon_keyword_freq", n=25))
    a(table("lexicon_keyword_freq", where=pl.col("n_sentences") == 0))
    a(f"### No-hit sentences: top n-grams (>= {st} tokens, stop words removed)\n")
    for k in (1, 2, 3):
        a(
            table(
                "nohit_ngrams",
                cols=["rank", "ngram", "count", "per_1k_nohit_sentences"],
                n=25,
                where=pl.col("n") == k,
            )
        )
    a("### Keyword quality check\n")
    a(
        "Five random example sentences per keyword; `[SUSPICIOUS]` marks a configured "
        "suspicious context (e.g. *short story*, *sounds like*); `suspicious_share` is over all "
        "hits of that keyword. Keywords with a suspicious share above 5%:\n"
    )
    a(
        table(
            "keyword_quality",
            cols=[
                "aspect",
                "keyword",
                "n_hits",
                "suspicious_share",
                "suspicious_contexts",
                "example_1",
                "example_2",
            ],
            where=pl.col("suspicious_share") > 0.05,
            sort=("suspicious_share", True),
        )
    )
    a("Ambiguous words (not matched) and how they are used:\n")
    a(
        table(
            "keyword_quality",
            cols=["keyword", "n_hits", "suspicious_share", "example_1", "example_2", "example_3"],
            where=pl.col("aspect").str.starts_with("(ambiguous"),
        )
    )
    a("All keywords with all five examples: `tables/keyword_quality.csv`.\n")
    a("### Per game\n")
    a(
        table(
            "lexicon_per_game_describe",
            cols=[
                "statistic",
                "hit_rate",
                "label_rate",
                "aspects_ge_5pct",
                "aspect_entropy",
                "short_sentence_share",
                "sentences",
            ],
        )
    )
    a(fig("lexicon_hit_rate_per_game", "Per-game hit rate"))

    a("## 4. Topic-model feasibility (system B) — purity is PRELIMINARY\n")
    a(
        f"Sample: up to {tc['sample_sentences']:,} sentences with >= {tc['min_tokens']} tokens, "
        f"equal quota per candidate game, seed {cfg['seed']}. "
        f"Embeddings `{tc['embedding_model']}`; "
        f"UMAP {tc['umap']} fitted once; HDBSCAN `min_cluster_size` in "
        f"{tc['hdbscan_min_cluster_sizes']} (10 = BERTopic default); c-TF-IDF with English stop "
        "words, min_df=5. Outlier reduction: BERTopic `reduce_outliers(strategy='c-tf-idf')` "
        "with default threshold 0 (assigns every outlier). Coherence: gensim c_v over the "
        "top-10 words, texts tokenized like the vectorizer.\n"
    )
    a(table("topic_runs"))
    a(fig("topic_runs", "Topic count and outlier share per setting"))
    a(f"Top 10 topics, `{main_cfg}`:\n")
    a(table("topic_top10", where=pl.col("config") == main_cfg))
    a("Top 10 topics, `mcs100_reduce-none`:\n")
    a(table("topic_top10", where=pl.col("config") == "mcs100_reduce-none"))
    a("### Topic → aspect purity (lexicon hits as weak labels)\n")
    a(
        f"`purity` = share of the topic's lexicon aspect hits that go to its top aspect; "
        f"`mixed` = purity < {tc['mixed_topic_max_purity']}; `low_coverage` = fewer than "
        f"{tc['low_coverage_max']:.0%} of its sentences have any lexicon aspect hit.\n"
    )
    a(table("topic_purity_summary"))
    a(fig("topic_purity_scatter", "Topic coverage vs purity"))
    a(f"Largest mixed topics (`{main_cfg}`):\n")
    a(
        table(
            "topic_aspect_purity",
            cols=[
                "topic",
                "size",
                "lexicon_coverage",
                "top_aspect",
                "purity",
                "second_aspect",
                "top_words",
            ],
            n=12,
            where=(pl.col("config") == main_cfg) & pl.col("mixed"),
            sort=("size", True),
        )
    )
    a(f"Largest low-coverage topics (`{main_cfg}`) — candidates for aspects the lexicon misses:\n")
    a(
        table(
            "topic_aspect_purity",
            cols=["topic", "size", "lexicon_coverage", "candidate_theme", "top_words", "example"],
            n=20,
            where=(pl.col("config") == main_cfg) & pl.col("low_coverage"),
            sort=("size", True),
        )
    )
    a("Candidate themes (topic top-words matched to `topics.candidate_themes` in the config):\n")
    a(table("topic_candidate_themes"))

    a("## 5. Game selection — PRELIMINARY (depends on the lexicon)\n")
    a(
        "Candidates: games with >= 200 clean English reviews. Score = weighted sum of z-scores "
        f"({', '.join(f'{k}×{v}' for k, v in cfg['selection']['weights'].items())}). "
        f"Steam metadata fetched for the top {cfg['selection']['shortlist']}; then greedy picks "
        f"with at least {cfg['selection']['min_negative_games']} games having >= "
        f"{cfg['selection']['negative_share_threshold']:.0%} negative reviews, at most "
        f"{cfg['selection']['max_per_primary_tag']} game per primary SteamSpy tag and "
        f"{cfg['selection']['max_per_store_genre']} per Steam genre. Genres come from the Steam "
        "store / SteamSpy, not inferred from names. Split: all 7/3 partitions searched, "
        "minimizing JS divergence of mean aspect profiles + ½ gap in recommend share + ½ gap in "
        "hit rate, with >= 1 negative-heavy game in test.\n"
    )
    sel = E.OUT_DIR / "selected_games.csv"
    if sel.is_file():
        s = pl.read_csv(sel)
        a("| game_id | name | split | justification |\n|---|---|---|---|")
        for r in s.iter_rows(named=True):
            a(f"| {r['game_id']} | {fmt(r['name'])} | {r['split']} | {fmt(r['justification'])} |")
        a("\n_Source: `selected_games.csv`_\n")
    a(
        table(
            "selected_games_metrics",
            cols=[
                "app_id",
                "name",
                "split",
                "primary_tag",
                "n_clean",
                "median_chars_clean",
                "hit_rate",
                "aspects_ge_5pct",
                "aspect_entropy",
                "informative_share",
                "pos_share_clean",
                "score",
            ],
        )
    )
    a(table("split_comparison"))
    a(
        f"Split objective (`tables/split_objective.json`): JS divergence "
        f"{split.get('js_divergence_bits', float('nan')):.4f} bits, recommend-share gap "
        f"{split.get('pos_share_gap', float('nan')):.3f}, hit-rate gap "
        f"{split.get('hit_rate_gap', float('nan')):.3f}, over "
        f"{split.get('combinations_checked', 0)} partitions.\n"
    )
    a(fig("split_comparison", "Train vs test aspect profile"))
    a("Top 20 scored candidates:\n")
    a(
        table(
            "game_candidates_scored",
            cols=[
                "app_id",
                "name",
                "primary_tag",
                "score",
                "n_clean",
                "hit_rate",
                "aspects_ge_5pct",
                "aspect_entropy",
                "informative_share",
                "pos_share_clean",
            ],
            n=20,
        )
    )

    a("## 6. Candidate labeling sample (not labeled, not committed)\n")
    a(
        f"`data/processed/candidate_sample.csv` and `.jsonl`: "
        f"{cfg['labeling_sample']['reviews_per_game']} clean English reviews per selected game, "
        f"uniform random from the game's full clean pool, seed {cfg['seed']}, same splitter. "
        "Columns: game_id, game_name, split, review_id (raw CSV row number), sentence_id "
        "(`<review_id>-<k>`), sentence, recommended, n_words. Workload assumes "
        f"{cfg['labeling_sample']['verify_seconds_per_sentence']} seconds to verify one LLM "
        "label.\n"
    )
    a(table("candidate_sample_summary"))

    out = E.OUT_DIR / "eda_report.md"
    out.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8", newline="\n")
    log.info("report -> %s (%d lines)", out, len(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
