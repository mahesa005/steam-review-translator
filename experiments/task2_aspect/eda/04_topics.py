"""Step 4: BERTopic feasibility (system B) on a stratified sentence sample.

Sample: sentences with >= topics.min_tokens word tokens from step 2, stratified across
candidate games (equal quota per game, seed 42), capped at topics.sample_sentences.
Embeddings (all-MiniLM-L6-v2) and the UMAP projection are computed once and cached in
artifacts/eda/; each HDBSCAN min_cluster_size is then fitted on the same projection, so the
comparison isolates the clustering setting. BERTopic is otherwise at defaults except the
c-TF-IDF vectorizer (English stop words, min_df=5), which only changes topic words.
"""

from __future__ import annotations

import math
import sys

import edalib as E
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def stratified_sample(cfg: dict, log) -> pl.DataFrame:
    tc = cfg["topics"]
    # sample on ids only, then fetch text for the picked rows (memory-safe)
    sents_lf = pl.scan_parquet(E.interim("sentences.parquet"))
    ids = (
        sents_lf.filter(pl.col("n_words") >= tc["min_tokens"])
        .select("review_id", "sent_idx", "app_id")
        .collect()
    )
    target = tc["sample_sentences"] if not E.SMOKE else 3000
    n_games = ids["app_id"].n_unique()
    quota = math.ceil(target / n_games)
    picked = (
        ids.with_columns(r=pl.int_range(pl.len()).shuffle(seed=cfg["seed"]).over("app_id"))
        .filter(pl.col("r") < quota)
        .sample(fraction=1.0, shuffle=True, seed=cfg["seed"])
        .head(target)
        .drop("r")
    )
    del ids
    keys = ["review_id", "sent_idx"]
    s = (
        sents_lf.select(*keys, "text", "n_words")
        .join(picked.lazy(), on=keys, how="inner")
        .join(
            pl.scan_parquet(E.interim("sentence_hits.parquet")).select(*keys, "aspects"),
            on=keys,
            how="left",
        )
        .collect()
        .sort("app_id", "review_id", "sent_idx")
    )
    log.info(
        "topic sample: %s sentences from %d games (quota %d/game)",
        f"{s.height:,}",
        s["app_id"].n_unique(),
        quota,
    )
    return s


def embed(docs: list[str], cfg: dict, log) -> np.ndarray:
    path = E.cache("topic_embeddings.npy")
    if path.is_file() and not E.FORCE:
        emb = np.load(path)
        if emb.shape[0] == len(docs):
            log.info("cached embeddings %s", emb.shape)
            return emb
    import torch
    from sentence_transformers import SentenceTransformer

    tc = cfg["topics"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = tc["batch_size_gpu"] if device == "cuda" else tc["batch_size_cpu"]
    log.info("embedding on %s (batch %d), torch threads=%d", device, bs, torch.get_num_threads())
    model = SentenceTransformer(tc["embedding_model"], device=device)
    with E.timed(log, f"embed {len(docs):,} sentences"):
        emb = model.encode(
            docs,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    np.save(path, emb)
    return emb


def reduce(emb: np.ndarray, cfg: dict, log) -> np.ndarray:
    path = E.cache("topic_umap.npy")
    if path.is_file() and not E.FORCE:
        red = np.load(path)
        if red.shape[0] == emb.shape[0]:
            log.info("cached UMAP %s", red.shape)
            return red
    from umap import UMAP

    u = cfg["topics"]["umap"]
    with E.timed(log, "UMAP"):
        red = UMAP(
            n_neighbors=u["n_neighbors"],
            n_components=u["n_components"],
            min_dist=u["min_dist"],
            metric=u["metric"],
            random_state=cfg["seed"],
        ).fit_transform(emb)
    np.save(path, red)
    return red


def coherence_cv(
    topic_words: list[list[str]], tokenized: list[list[str]], dictionary, processes: int
) -> list[float]:
    from gensim.models.coherencemodel import CoherenceModel

    words = [[w for w in ws if w in dictionary.token2id] for ws in topic_words]
    keep = [i for i, ws in enumerate(words) if len(ws) >= 2]
    if not keep:
        return [float("nan")] * len(words)
    cm = CoherenceModel(
        topics=[words[i] for i in keep],
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
        processes=processes,
    )
    scores = cm.get_coherence_per_topic()
    out = [float("nan")] * len(words)
    for i, sc in zip(keep, scores, strict=True):
        out[i] = float(sc)
    return out


def theme_of(words: list[str], themes: dict[str, list[str]]) -> str:
    hit = [t for t, kws in themes.items() if any(w in kws for w in words[:10])]
    return ",".join(hit)


def main() -> int:
    cfg = E.load_cfg()
    log = E.logger("04_topics")
    tc = cfg["topics"]
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from gensim.corpora import Dictionary
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    sample = stratified_sample(cfg, log)
    docs = sample["text"].to_list()
    emb = embed(docs, cfg, log)
    red = reduce(emb, cfg, log)

    vec_cfg = tc["vectorizer"]
    min_df = vec_cfg["min_df"] if not E.SMOKE else 2

    def vectorizer() -> CountVectorizer:
        return CountVectorizer(stop_words=vec_cfg["stop_words"], min_df=min_df)

    analyzer = vectorizer().build_analyzer()
    tokenized = [analyzer(d) for d in docs]
    dictionary = Dictionary(tokenized)
    aspects = sample["aspects"].to_list()
    themes = tc["candidate_themes"]

    runs, top_rows, purity_rows = [], [], []
    assignments: dict[str, list[int]] = {}
    sizes = tc["hdbscan_min_cluster_sizes"] if not E.SMOKE else [10, 30]
    for mcs in sizes:
        hdb = HDBSCAN(
            min_cluster_size=mcs,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        model = BERTopic(
            umap_model=BaseDimensionalityReduction(),
            hdbscan_model=hdb,
            vectorizer_model=vectorizer(),
            top_n_words=tc["top_n_words"],
            calculate_probabilities=False,
            verbose=False,
        )
        with E.timed(log, f"BERTopic min_cluster_size={mcs}"):
            topics, _ = model.fit_transform(docs, embeddings=red)
        for red_name in ("none", "c-tf-idf"):
            if red_name == "none":
                tlist = list(topics)
            else:
                with E.timed(log, f"reduce_outliers (c-tf-idf) mcs={mcs}"):
                    tlist = list(model.reduce_outliers(docs, topics, strategy="c-tf-idf"))
                    model.update_topics(
                        docs,
                        topics=tlist,
                        vectorizer_model=vectorizer(),
                        top_n_words=tc["top_n_words"],
                    )
            m = model
            cfg_name = f"mcs{mcs}_reduce-{red_name}"
            assignments[cfg_name] = tlist
            tids = sorted({t for t in tlist if t != -1})
            words = {t: [w for w, _ in (m.get_topic(t) or [])][: tc["top_n_words"]] for t in tids}
            with E.timed(log, f"c_v coherence {cfg_name} ({len(tids)} topics)"):
                cv = coherence_cv(
                    [words[t] for t in tids],
                    tokenized,
                    dictionary,
                    1 if E.SMOKE else tc["coherence_processes"],
                )
            cv_by = dict(zip(tids, cv, strict=True))
            arr = np.array(tlist)
            valid_cv = [c for c in cv if not math.isnan(c)]
            runs.append(
                {
                    "config": cfg_name,
                    "min_cluster_size": mcs,
                    "reduce_outliers": red_name,
                    "n_topics": len(tids),
                    "outlier_share": float((arr == -1).mean()),
                    "cv_mean": float(np.mean(valid_cv)) if valid_cv else None,
                    "cv_median": float(np.median(valid_cv)) if valid_cv else None,
                    "largest_topic_share": float(max((arr == t).mean() for t in tids))
                    if tids
                    else None,
                }
            )
            counts = {t: int((arr == t).sum()) for t in tids}
            for rank, t in enumerate(sorted(tids, key=lambda x: -counts[x])[:10], 1):
                top_rows.append(
                    {
                        "config": cfg_name,
                        "rank": rank,
                        "topic": t,
                        "size": counts[t],
                        "share": counts[t] / len(arr),
                        "cv": cv_by[t],
                        "top_words": " ".join(words[t]),
                    }
                )
            if red_name != "none":
                continue
            # topic -> aspect purity using lexicon hits as weak labels
            for t in tids:
                members = np.flatnonzero(arr == t)
                labels = [a for i in members for a in (aspects[i] or []) if a != "overall"]
                hit_sents = sum(
                    1 for i in members if any(a != "overall" for a in (aspects[i] or []))
                )
                c = {a: labels.count(a) for a in set(labels)}
                tot = sum(c.values())
                ranked = sorted(c.items(), key=lambda kv: -kv[1])
                purity = ranked[0][1] / tot if tot else 0.0
                coverage = hit_sents / len(members)
                purity_rows.append(
                    {
                        "config": cfg_name,
                        "topic": t,
                        "size": len(members),
                        "lexicon_coverage": coverage,
                        "top_aspect": ranked[0][0] if ranked else "",
                        "purity": purity,
                        "second_aspect": ranked[1][0] if len(ranked) > 1 else "",
                        "second_share": ranked[1][1] / tot if len(ranked) > 1 else 0.0,
                        "mixed": bool(tot and purity < tc["mixed_topic_max_purity"]),
                        "low_coverage": coverage < tc["low_coverage_max"],
                        "candidate_theme": theme_of(words[t], themes),
                        "cv": cv_by[t],
                        "top_words": " ".join(words[t]),
                        "example": E.normalize_ws(docs[members[0]])[:140].rstrip(),
                    }
                )

    runs_df = pl.DataFrame(runs)
    E.save_table(runs_df, "topic_runs", log)
    E.save_table(pl.DataFrame(top_rows), "topic_top10", log)
    purity = pl.DataFrame(purity_rows, infer_schema_length=None)
    E.save_table(purity, "topic_aspect_purity", log)

    summ = (
        purity.group_by("config")
        .agg(
            topics=pl.len(),
            mixed_share=pl.col("mixed").mean(),
            low_coverage_share=pl.col("low_coverage").mean(),
            sentences_in_low_coverage_topics=pl.col("size").filter("low_coverage").sum(),
            size_weighted_purity=(pl.col("purity") * pl.col("size")).sum() / pl.col("size").sum(),
        )
        .sort("config")
    )
    E.save_table(summ, "topic_purity_summary", log)
    theme_rows = []
    for th in themes:
        sub = purity.filter(pl.col("candidate_theme").str.contains(th))
        theme_rows.append(
            {
                "theme": th,
                "topics": sub.height,
                "low_coverage_topics": sub.filter("low_coverage").height,
                "sentences": int(sub["size"].sum()) if sub.height else 0,
                "configs": ",".join(sorted(set(sub["config"].to_list()))),
            }
        )
    E.save_table(
        pl.DataFrame(theme_rows).sort("sentences", descending=True), "topic_candidate_themes", log
    )

    main_cfg = f"mcs{sizes[0]}_reduce-none"
    sample.select("review_id", "sent_idx", "app_id").with_columns(
        **{k: pl.Series(v) for k, v in assignments.items()}
    ).write_parquet(E.interim("topic_assignments.parquet"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    lab = runs_df["config"].to_list()
    ax[0].barh(lab, runs_df["outlier_share"].to_numpy(), color="#c44e52")
    ax[0].set_xlabel("outlier share (topic -1)")
    ax[1].barh(lab, runs_df["n_topics"].to_numpy(), color="#4c72b0")
    ax[1].set_xlabel("number of topics")
    E.save_fig(fig, "topic_runs", log)
    p_main = purity.filter(pl.col("config") == main_cfg)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        p_main["lexicon_coverage"].to_numpy(),
        p_main["purity"].to_numpy(),
        s=np.sqrt(p_main["size"].to_numpy()) * 2,
        alpha=0.5,
    )
    ax.axvline(tc["low_coverage_max"], ls="--", c="k", lw=1)
    ax.axhline(tc["mixed_topic_max_purity"], ls="--", c="k", lw=1)
    ax.set_xlabel("lexicon coverage of topic")
    ax.set_ylabel("aspect purity")
    ax.set_title(f"Topic vs lexicon aspects ({main_cfg}, PRELIMINARY)")
    E.save_fig(fig, "topic_purity_scatter", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
