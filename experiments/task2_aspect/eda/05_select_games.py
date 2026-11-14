"""Step 5: score candidate games, fetch Steam metadata for a shortlist, pick 10 (+3 backups),
and propose a 7 train / 3 test split with comparable aspect distributions.

Genres come from the Steam store (appdetails) and SteamSpy tags, fetched only for the
shortlist (rate limits) and cached in data/interim/task2_eda/steam_meta/.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
import urllib.error
import urllib.request

import edalib as E
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

STORE = "https://store.steampowered.com/api/appdetails?appids={}&l=english"
SPY = "https://steamspy.com/api.php?request=appdetails&appid={}"


def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "IF5153-group17-eda/0.1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(5 * (attempt + 1))
    return None


def fetch_meta(app_ids: list[int], cfg: dict, log) -> pl.DataFrame:
    d = E.interim("steam_meta").with_suffix("")
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, app in enumerate(app_ids):
        path = d / f"{app}.json"
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
        else:
            store = _get(STORE.format(app)) or {}
            time.sleep(cfg["selection"]["steam_request_interval_s"])
            spy = _get(SPY.format(app)) or {}
            entry = store.get(str(app), {})
            data = entry.get("data", {}) if entry.get("success") else {}
            meta = {
                "store_ok": bool(data),
                "name": data.get("name") or spy.get("name") or "",
                "type": data.get("type", ""),
                "genres": [g["description"] for g in data.get("genres", [])],
                "release": (data.get("release_date") or {}).get("date", ""),
                "tags": list((spy.get("tags") or {}).keys())[:10]
                if isinstance(spy.get("tags"), dict)
                else [],
            }
            path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            if (i + 1) % 25 == 0:
                log.info("  steam meta %d/%d", i + 1, len(app_ids))
        generic = set(cfg["selection"]["generic_tags"])
        primary = next(
            (t for t in meta["tags"] if t not in generic),
            meta["tags"][0] if meta["tags"] else (meta["genres"][0] if meta["genres"] else ""),
        )
        rows.append(
            {
                "app_id": app,
                "name": meta["name"],
                "type": meta["type"] or "unknown",
                "store_listed": meta["store_ok"],
                "genres": "|".join(meta["genres"]),
                "tags": "|".join(meta["tags"][:5]),
                "primary_tag": primary,
                "release": meta["release"],
            }
        )
    return pl.DataFrame(rows)


def zscore(s: pl.Series) -> pl.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def justification(r: dict) -> str:
    shares = sorted(((a, r[f"share_{a}"]) for a in E.ASPECTS), key=lambda kv: -kv[1])
    top = ", ".join(f"{a} {s:.0%}" for a, s in shares[:3])
    genre = r["primary_tag"] or r["genres"].split("|")[0] or "genre unknown"
    return (
        f"{genre}; {r['n_clean']:,} clean EN reviews, {1 - r['pos_share_clean']:.0%} negative; "
        f"lexicon hit rate {r['hit_rate']:.0%}, {r['aspects_ge_5pct']} aspects >=5% "
        f"(top: {top})"
    )


def main() -> int:
    cfg = E.load_cfg()
    log = E.logger("05_select_games")
    sc = cfg["selection"]
    w = sc["weights"]

    games = pl.read_csv(E.TABLES_DIR / "game_stats.csv")
    lex = pl.read_csv(E.TABLES_DIR / "lexicon_per_game.csv")
    c = lex.join(games, on="app_id", how="inner").filter(
        pl.col("n_clean") >= cfg["candidates"]["min_english_reviews"]
    )
    c = c.with_columns(
        informative_share=1 - (pl.col("short_share_en") + pl.col("spam_share_en")).clip(0, 1),
        log_n_clean=pl.col("n_clean").log(),
        negative_share=1 - pl.col("pos_share_clean"),
    ).with_columns(negative_share_capped=pl.col("negative_share").clip(0, 0.5))
    c = c.with_columns(score=sum(wt * zscore(c[m]) for m, wt in w.items())).sort(
        "score", descending=True
    )
    log.info("%d candidate games scored", c.height)

    shortlist = c.head(sc["shortlist"] if not E.SMOKE else 15)["app_id"].to_list()
    with E.timed(log, f"steam metadata for {len(shortlist)} shortlisted games"):
        meta = fetch_meta(shortlist, cfg, log)
    scored = c.join(meta, on="app_id", how="left")
    E.save_table(
        scored.select(
            "app_id",
            "name",
            "type",
            "primary_tag",
            "genres",
            "score",
            "n_clean",
            "median_chars_clean",
            "hit_rate",
            "aspects_ge_5pct",
            "aspect_entropy",
            "informative_share",
            "short_share_en",
            "spam_share_en",
            "pos_share_clean",
            "helpful_share_clean",
            "sentences",
            *[f"share_{a}" for a in E.ASPECTS],
        ),
        "game_candidates_scored",
        log,
    )

    # ---------------- greedy diverse selection
    pool = (
        scored.filter(pl.col("app_id").is_in(shortlist))
        .filter((pl.col("name") != "") & pl.col("type").is_in(["game", "unknown"]))
        .sort("score", descending=True)
        .to_dicts()
    )
    n_total = sc["n_games"] + sc["n_backups"]
    picked: list[dict] = []

    def ok(r: dict, chosen: list[dict]) -> bool:
        if sum(x["primary_tag"] == r["primary_tag"] for x in chosen) >= sc["max_per_primary_tag"]:
            return False
        gs = set(filter(None, r["genres"].split("|")))
        return all(
            sum(g in x["genres"].split("|") for x in chosen) < sc["max_per_store_genre"] for g in gs
        )

    neg_thr = sc["negative_share_threshold"]
    for r in pool:  # reserve slots for games with real criticism first
        if sum(x["negative_share"] >= neg_thr for x in picked) >= sc["min_negative_games"]:
            break
        if r["negative_share"] >= neg_thr and ok(r, picked):
            picked.append(r)
    for r in pool:
        if len(picked) >= n_total:
            break
        if r not in picked and ok(r, picked):
            picked.append(r)
    if len(picked) < n_total:  # relax genre constraints if the shortlist is too uniform
        log.warning("relaxing genre constraints: only %d diverse games found", len(picked))
        for r in pool:
            if len(picked) >= n_total:
                break
            if r not in picked:
                picked.append(r)
    picked.sort(key=lambda r: -r["score"])
    # keep the negative-heavy reserved games inside the main 10
    main = [r for r in picked if r["negative_share"] >= neg_thr][: sc["min_negative_games"]]
    main += [r for r in picked if r not in main][: sc["n_games"] - len(main)]
    backups = [r for r in picked if r not in main][: sc["n_backups"]]
    main.sort(key=lambda r: -r["score"])

    # ---------------- 7/3 split with comparable aspect distributions
    def profile(rs: list[dict]) -> np.ndarray:
        return np.mean([[r[f"share_{a}"] for a in E.ASPECTS] for r in rs], axis=0)

    best = None
    n_test = len(main) - sc["n_train"]
    for test_idx in itertools.combinations(range(len(main)), n_test):
        test = [main[i] for i in test_idx]
        train = [r for i, r in enumerate(main) if i not in test_idx]
        if not any(r["negative_share"] >= neg_thr for r in test):
            continue
        jsd = js_divergence(profile(train) + 1e-9, profile(test) + 1e-9)
        pos_gap = abs(
            np.mean([r["pos_share_clean"] for r in train])
            - np.mean([r["pos_share_clean"] for r in test])
        )
        hit_gap = abs(
            np.mean([r["hit_rate"] for r in train]) - np.mean([r["hit_rate"] for r in test])
        )
        obj = jsd + 0.5 * pos_gap + 0.5 * hit_gap
        if best is None or obj < best[0]:
            best = (obj, test_idx, jsd, pos_gap, hit_gap)
    assert best is not None
    test_ids = {main[i]["app_id"] for i in best[1]}
    E.save_json(
        {
            "objective": best[0],
            "js_divergence_bits": best[2],
            "pos_share_gap": best[3],
            "hit_rate_gap": best[4],
            "combinations_checked": math.comb(len(main), n_test),
        },
        "split_objective",
    )

    sel_rows = []
    for r in main:
        sel_rows.append(
            {
                "game_id": r["app_id"],
                "name": r["name"],
                "split": "test" if r["app_id"] in test_ids else "train",
                "justification": justification(r),
            }
        )
    for r in backups:
        sel_rows.append(
            {
                "game_id": r["app_id"],
                "name": r["name"],
                "split": "backup",
                "justification": justification(r),
            }
        )
    sel = pl.DataFrame(sel_rows)
    out = E.OUT_DIR / "selected_games.csv"
    sel.write_csv(out)
    log.info("selected games -> %s", out)
    metrics = (
        scored.join(sel.rename({"game_id": "app_id"}).select("app_id", "split"), on="app_id")
        .select(
            "app_id",
            "name",
            "split",
            "primary_tag",
            "genres",
            "score",
            "n_clean",
            "median_chars_clean",
            "hit_rate",
            "aspects_ge_5pct",
            "aspect_entropy",
            "informative_share",
            "pos_share_clean",
            *[f"share_{a}" for a in E.ASPECTS],
        )
        .sort("split", "score", descending=[False, True])
    )
    E.save_table(metrics, "selected_games_metrics", log)

    tr = [r for r in main if r["app_id"] not in test_ids]
    te = [r for r in main if r["app_id"] in test_ids]
    comp = [
        {"metric": f"share_{a}", "train": float(x), "test": float(y)}
        for a, x, y in zip(E.ASPECTS, profile(tr), profile(te), strict=True)
    ]
    for m in ("hit_rate", "pos_share_clean", "aspect_entropy", "median_chars_clean"):
        comp.append(
            {
                "metric": m,
                "train": float(np.mean([r[m] for r in tr])),
                "test": float(np.mean([r[m] for r in te])),
            }
        )
    E.save_table(pl.DataFrame(comp), "split_comparison", log)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(E.ASPECTS))
    ax.bar(x - 0.2, profile(tr), 0.4, label=f"train ({len(tr)} games)")
    ax.bar(x + 0.2, profile(te), 0.4, label=f"test ({len(te)} games)")
    ax.set_xticks(x, E.ASPECTS, rotation=30)
    ax.set_ylabel("mean share of aspect hits")
    ax.legend()
    ax.set_title("Train vs test aspect profile (PRELIMINARY lexicon)")
    E.save_fig(fig, "split_comparison", log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
