"""Step 3: lexicon hit rate (feasibility of system A). PRELIMINARY: placeholder lexicon.

Matching = library whole-token LexiconMatcher run on lowercase surface tokens AND spaCy
lemmas (union). Ambiguous words never match on their own.
"""

from __future__ import annotations

import math
import random
import re
import subprocess
import sys
from collections import Counter

import edalib as E
import matplotlib.pyplot as plt
import polars as pl
import pyarrow.parquet as pq

from steam_review_translator.aspect.lexicon import tokenize
from steam_review_translator.core.config import project_root


def lexicon_version(path: str) -> dict:
    full = project_root() / path
    try:
        blob = subprocess.run(
            ["git", "hash-object", str(full)], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        blob = ""
    return {
        "path": path,
        "sha256": E.sha256(full),
        "git_blob": blob,
        "status": "PRELIMINARY placeholder seed lexicon",
    }


def entropy(shares: list[float]) -> float:
    """Normalized Shannon entropy in [0, 1] over the given categories."""
    nz = [s for s in shares if s > 0]
    if len(shares) < 2:
        return 0.0
    return -sum(s * math.log(s) for s in nz) / math.log(len(shares))


def ngrams(seq: list[str], n: int) -> list[str]:
    return [" ".join(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def main() -> int:
    cfg = E.load_cfg()
    log = E.logger("03_lexicon")
    lc = cfg["lexicon"]
    rng = random.Random(cfg["seed"])
    version = lexicon_version(lc["path"])
    E.save_json(version, "lexicon_version")
    log.info("lexicon %s sha256=%s", lc["path"], version["sha256"][:12])

    from spacy.lang.en.stop_words import STOP_WORDS

    stop = set(STOP_WORDS) | {
        "game", "games", "play", "playing", "played", "get", "like",
        "s", "nt", "'s", "n't", "'m", "'re", "'ve", "'ll", "'d",
    }  # fmt: skip
    informative = cfg["sentences"]["short_sentence_tokens"]
    k_ex = lc["examples_per_keyword"]
    susp_cfg = {k: [p.lower() for p in v] for k, v in lc["suspicious_contexts"].items()}
    amb = sorted(E.load_lexicon_ambiguous(lc["path"]))
    amb_re = re.compile(r"\b(" + "|".join(map(re.escape, amb)) + r")\b", re.I)

    # One chunked pass over the sentences (memory-safe): match (or reuse cached hits),
    # write hits incrementally, and accumulate the text-based statistics as we go.
    # Examples are drawn by reservoir sampling (seed 42), so each keyword gets a uniform
    # random sample of k sentences without holding all texts in memory.
    sents_lf = pl.scan_parquet(E.interim("sentences.parquet"))
    n_total = sents_lf.select(pl.len()).collect().item()
    hits_path = E.interim("sentence_hits.parquet")
    cached = E.is_cached(hits_path, log)
    hits_lf = pl.scan_parquet(hits_path) if cached else None
    matcher = None
    tmp_path = hits_path.with_suffix(".partial.parquet")
    if not cached:
        nlp = E.load_nlp(cfg["sentences"]["spacy_model"])
        matcher = E.LemmaLexiconMatcher(nlp, lc["path"])
    writer = None
    counters: dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter()}
    n_nohit = 0
    kw_stats: dict[tuple[str, str], dict] = {}
    amb_stats: dict[str, dict] = {w: {"n": 0, "susp": 0, "res": []} for w in amb}

    def reservoir(stat: dict, text: str) -> None:
        stat["n"] += 1
        if len(stat["res"]) < k_ex:
            stat["res"].append(text)
        else:
            j = rng.randrange(stat["n"])
            if j < k_ex:
                stat["res"][j] = text

    chunk_rows = int(lc.get("chunk_rows", 250_000))
    with E.timed(log, f"lexicon pass over {n_total:,} sentences (chunks of {chunk_rows:,})"):
        for off in range(0, n_total, chunk_rows):
            chunk = (
                sents_lf.slice(off, chunk_rows)
                .select("review_id", "sent_idx", "app_id", "text", "lemmas", "n_words")
                .collect()
            )
            if matcher is not None:
                found = [
                    matcher.match_terms(t, lem)
                    for t, lem in zip(chunk["text"], chunk["lemmas"], strict=True)
                ]
                part = chunk.select("review_id", "sent_idx", "app_id").with_columns(
                    aspects=pl.Series(
                        [sorted({a for a, _ in h}) for h in found], dtype=pl.List(pl.Utf8)
                    ),
                    keywords=pl.Series(
                        [[f"{a}:{k}" for a, k in h] for h in found], dtype=pl.List(pl.Utf8)
                    ),
                )
                table = part.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
                writer.write_table(table)
            else:
                assert hits_lf is not None
                part = hits_lf.slice(off, chunk_rows).select("aspects", "keywords").collect()
            kws = part["keywords"].to_list()
            labels = part["aspects"].to_list()
            for text, lem, nw, kw_list, lab in zip(
                chunk["text"], chunk["lemmas"], chunk["n_words"], kws, labels, strict=True
            ):
                low = text.lower()
                for item in kw_list:
                    aspect, keyword = item.split(":", 1)
                    st = kw_stats.setdefault((aspect, keyword), {"n": 0, "susp": 0, "res": []})
                    is_susp = any(c in low for c in susp_cfg.get(keyword, []))
                    st["susp"] += is_susp
                    reservoir(
                        st,
                        ("[SUSPICIOUS] " if is_susp else "") + E.normalize_ws(text)[:160].rstrip(),
                    )
                for w in {m.lower() for m in amb_re.findall(text)}:
                    st = amb_stats[w]
                    st["susp"] += any(c in low for c in susp_cfg.get(w, []))
                    reservoir(st, E.normalize_ws(text)[:160].rstrip())
                if not lab and nw >= informative:
                    n_nohit += 1
                    toks = tokenize(lem)
                    counters[1].update(t for t in toks if t not in stop and re.search(r"[a-z]", t))
                    for k in (2, 3):
                        counters[k].update(
                            g
                            for g in ngrams(toks, k)
                            if not any(t in stop or not re.search(r"[a-z]", t) for t in g.split())
                        )
            log.info("  %s / %s sentences", f"{min(off + chunk_rows, n_total):,}", f"{n_total:,}")
            del chunk, part
    if writer is not None:
        writer.close()
        tmp_path.replace(hits_path)

    ng_rows = []
    for k, c in counters.items():
        for rank, (g, cnt) in enumerate(c.most_common(lc["ngram_top_k"]), 1):
            ng_rows.append(
                {
                    "n": k,
                    "rank": rank,
                    "ngram": g,
                    "count": cnt,
                    "per_1k_nohit_sentences": 1000 * cnt / max(1, n_nohit),
                }
            )
    E.save_table(pl.DataFrame(ng_rows), "nohit_ngrams", log)

    def kq_row(aspect: str, keyword: str, st: dict) -> dict:
        ctx = susp_cfg.get(keyword, [])
        return {
            "aspect": aspect,
            "keyword": keyword,
            "n_hits": st["n"],
            "suspicious_share": st["susp"] / st["n"] if ctx and st["n"] else None,
            "suspicious_contexts": "; ".join(ctx),
            **{f"example_{j + 1}": st["res"][j] if j < len(st["res"]) else "" for j in range(k_ex)},
        }

    kq_rows = [kq_row(a, k, st) for (a, k), st in sorted(kw_stats.items())]
    kq_rows += [kq_row("(ambiguous, not matched)", w, amb_stats[w]) for w in amb]
    E.save_table(pl.DataFrame(kq_rows, infer_schema_length=None), "keyword_quality", log)

    # Light frame for the numeric metrics: ids + labels + n_words only (no text).
    df = (
        pl.scan_parquet(hits_path)
        .select("review_id", "sent_idx", "app_id", "aspects", "keywords")
        .join(sents_lf.select("review_id", "sent_idx", "n_words"), on=["review_id", "sent_idx"])
        .collect()
        .with_columns(
            n_labels=pl.col("aspects").list.len(),
            n_aspects=pl.col("aspects").list.set_difference(pl.lit(["overall"])).list.len(),
            informative=pl.col("n_words") >= informative,
        )
    )
    n = df.height

    # ---------------- overall hit rate
    rows = []
    for pop, d in (
        ("all_sentences", df),
        (f"sentences_ge_{informative}_tokens", df.filter("informative")),
    ):
        rows += [
            {"population": pop, "metric": "sentences", "value": d.height},
            {
                "population": pop,
                "metric": "share with >=1 label (9 aspects + overall)",
                "value": (d["n_labels"] > 0).mean(),
            },
            {
                "population": pop,
                "metric": "share with >=1 of the 9 aspects",
                "value": (d["n_aspects"] > 0).mean(),
            },
            {
                "population": pop,
                "metric": "share with NO hit (implicit/other risk)",
                "value": (d["n_labels"] == 0).mean(),
            },
            {
                "population": pop,
                "metric": "share with >=2 aspects",
                "value": (d["n_aspects"] >= 2).mean(),
            },
            {
                "population": pop,
                "metric": "review-level share with >=1 aspect hit",
                "value": d.group_by("review_id")
                .agg((pl.col("n_aspects") > 0).any())["n_aspects"]
                .mean(),
            },
        ]
    E.save_table(pl.DataFrame(rows), "lexicon_hit_rate", log)

    per_aspect = (
        df.select("aspects")
        .explode("aspects")
        .drop_nulls()
        .group_by("aspects")
        .agg(pl.len().alias("n_sentences"))
        .rename({"aspects": "aspect"})
        .with_columns(
            share_of_sentences=pl.col("n_sentences") / n,
            share_of_hits=pl.col("n_sentences") / pl.col("n_sentences").sum(),
        )
        .sort("n_sentences", descending=True)
    )
    E.save_table(per_aspect, "lexicon_hits_per_aspect", log)
    aps = (
        df.group_by("n_aspects")
        .agg(pl.len().alias("n_sentences"))
        .with_columns(share=pl.col("n_sentences") / n)
        .sort("n_aspects")
    )
    E.save_table(aps, "aspects_per_sentence", log)

    kw = (
        df.select("keywords")
        .explode("keywords")
        .drop_nulls()
        .group_by("keywords")
        .agg(pl.len().alias("n_sentences"))
        .with_columns(
            pl.col("keywords").str.split_exact(":", 1).struct.rename_fields(["aspect", "keyword"])
        )
        .unnest("keywords")
        .sort("n_sentences", descending=True)
    )
    lex = E.load_lexicon_terms(lc["path"])
    all_kw = pl.DataFrame([{"aspect": a, "keyword": k} for a, ks in lex.items() for k in ks])
    kw = (
        all_kw.join(kw, on=["aspect", "keyword"], how="left")
        .with_columns(pl.col("n_sentences").fill_null(0))
        .sort("n_sentences", descending=True)
    )
    E.save_table(kw, "lexicon_keyword_freq", log)

    # ---------------- per game
    game_rows = []
    for (app_id,), g in df.group_by(["app_id"]):
        cnt = Counter(a for lst in g["aspects"] for a in lst)
        asp_total = sum(cnt[a] for a in E.ASPECTS)
        shares = [cnt[a] / asp_total if asp_total else 0.0 for a in E.ASPECTS]
        inf = g.filter("informative")
        game_rows.append(
            {
                "app_id": app_id,
                "sample_reviews": g["review_id"].n_unique(),
                "sentences": g.height,
                "hit_rate": (g["n_aspects"] > 0).mean(),
                "label_rate": (g["n_labels"] > 0).mean(),
                "hit_rate_informative": (inf["n_aspects"] > 0).mean() if inf.height else None,
                "short_sentence_share": 1 - g["informative"].mean(),
                "aspects_ge_5pct": sum(s >= 0.05 for s in shares),
                "aspect_entropy": entropy(shares),
                **{f"share_{a}": s for a, s in zip(E.ASPECTS, shares, strict=True)},
                "overall_rate": cnt["overall"] / g.height,
            }
        )
    per_game = pl.DataFrame(game_rows).sort("app_id")
    E.save_table(per_game, "lexicon_per_game", log)
    E.save_table(per_game.drop("app_id").describe(), "lexicon_per_game_describe", log)

    # ---------------- figures
    pa_ = per_aspect.sort("n_sentences")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(pa_["aspect"].to_list(), pa_["share_of_sentences"].to_numpy(), color="#4c72b0")
    ax.set_xlabel("share of sentences with a hit")
    ax.set_title("Lexicon hits per aspect (PRELIMINARY)")
    E.save_fig(fig, "lexicon_hits_per_aspect", log)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_game["hit_rate"].to_numpy(), bins=40, color="#55a868")
    ax.set_xlabel("sentence hit rate (9 aspects) per game")
    ax.set_ylabel("games")
    ax.set_title("Per-game lexicon hit rate (PRELIMINARY)")
    E.save_fig(fig, "lexicon_hit_rate_per_game", log)
    log.info("overall 9-aspect hit rate: %.3f", (df["n_aspects"] > 0).mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
