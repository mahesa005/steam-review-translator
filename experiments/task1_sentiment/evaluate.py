"""Evaluate one Task 1 system on Metacritic test and Steam (review + distribution level).

    python experiments/task1_sentiment/evaluate.py --system vader
    python experiments/task1_sentiment/evaluate.py --system tfidf \
        --model artifacts/models/task1/s1_tfidf.pkl
    python experiments/task1_sentiment/evaluate.py --system transformer --name S2-zeroshot \
        --model cardiffnlp/twitter-roberta-base-sentiment-latest
    python experiments/task1_sentiment/evaluate.py --system transformer --name S4-roberta \
        --model artifacts/models/task1/roberta-base_len256

Writes to artifacts/runs/task1/<name>/:
  predictions_{metacritic_val,metacritic_test,steam}.parquet  per-review scores + labels
  metrics.json                                                every number below

Reported:
- review level: accuracy, macro-F1, per-class F1, confusion matrix
  (Metacritic: 3 classes; Steam: binary after collapsing pos vs neg)
- distribution level per game: MAE, TVD, dominant-label agreement, signed bias, Pearson
- ACC variant: counts corrected with the confusion matrix estimated on Metacritic val
- batch-size ablation: TVD of the predicted distribution from random batches of
  10/50/100/200 reviews vs the game's full gold distribution, plus the sampling-only
  TVD (gold of the batch vs full gold) to separate sampling noise from classifier error
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _common import write_json

from steam_review_translator.core.config import get_path, load_yaml
from steam_review_translator.core.schemas import Polarity
from steam_review_translator.sentiment.base import ReviewClassifier
from steam_review_translator.sentiment.labels import BINARY_CLASSES, CLASSES, collapse_to_binary
from steam_review_translator.sentiment.metrics import (
    adjusted_classify_and_count,
    classification_report,
    confusion_matrix,
    distribution_report,
    misclassification_rates,
    total_variation,
)

BATCH_SIZES = (10, 50, 100, 200)


def build_system(args: argparse.Namespace) -> ReviewClassifier:
    if args.system == "vader":
        from steam_review_translator.sentiment.vader import VaderSentimentAnalyzer

        return VaderSentimentAnalyzer()
    if args.system == "tfidf":
        from steam_review_translator.sentiment.tfidf import TfidfSentimentAnalyzer

        return TfidfSentimentAnalyzer(model_path=args.model)
    from steam_review_translator.sentiment.transformer import TransformerSentimentAnalyzer

    return TransformerSentimentAnalyzer(
        model=args.model, max_length=args.max_length, batch_size=args.batch_size
    )


def predict(clf: ReviewClassifier, df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Score every review once (cached on disk) and derive 3-class + binary labels."""
    if path.is_file():
        cached = pd.read_parquet(path)
        if len(cached) == len(df):
            return cached
    t0 = time.time()
    texts = df.text.tolist()
    scores = []
    step = 2048
    for start in range(0, len(texts), step):
        scores += clf.scores(texts[start : start + step])
        print(
            f"    {min(start + step, len(texts)):,}/{len(texts):,} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    labels = [clf.label_from_scores(s) for s in scores]
    out = df.copy()
    for c in CLASSES:
        out[f"score_{c.value}"] = [s.get(c, 0.0) for s in scores]
    out["pred"] = [lab.value for lab in labels]
    out["pred_binary"] = [collapse_to_binary(s).value for s in scores]
    out.to_parquet(path, index=False)
    return out


def shares_by(df: pd.DataFrame, key: str, col: str, classes: tuple[Polarity, ...]) -> pd.DataFrame:
    counts = pd.crosstab(df[key], df[col]).reindex(columns=[c.value for c in classes], fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def distribution_block(
    df: pd.DataFrame,
    key: str,
    gold_col: str,
    pred_col: str,
    classes: tuple[Polarity, ...],
    rates: list[list[float]] | None,
) -> dict[str, Any]:
    gold = shares_by(df, key, gold_col, classes)
    pred = shares_by(df, key, pred_col, classes).reindex(gold.index, fill_value=0.0)
    block: dict[str, Any] = {
        "cc": distribution_report(gold.values.tolist(), pred.values.tolist(), classes)
    }
    if rates is not None:
        adjusted = [adjusted_classify_and_count(p, rates) for p in pred.values.tolist()]
        block["acc"] = distribution_report(gold.values.tolist(), adjusted, classes)
    block["per_game"] = [
        {
            "game": str(g),
            "gold": gold.loc[g].round(4).tolist(),
            "pred": pred.loc[g].round(4).tolist(),
        }
        for g in gold.index
    ]
    return block


def batch_ablation(
    df: pd.DataFrame,
    key: str,
    gold_col: str,
    pred_col: str,
    classes: tuple[Polarity, ...],
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    names = [c.value for c in classes]
    result: dict[str, Any] = {}
    for size in BATCH_SIZES:
        model_tvd: list[float] = []
        sampling_tvd: list[float] = []
        for _, g in df.groupby(key):
            if len(g) < size:
                continue
            full = [float((g[gold_col] == n).mean()) for n in names]
            gold_arr, pred_arr = g[gold_col].to_numpy(), g[pred_col].to_numpy()
            for _ in range(repeats):
                idx = rng.choice(len(g), size=size, replace=False)
                pred = [float((pred_arr[idx] == n).mean()) for n in names]
                sub_gold = [float((gold_arr[idx] == n).mean()) for n in names]
                model_tvd.append(total_variation(full, pred))
                sampling_tvd.append(total_variation(full, sub_gold))
        if model_tvd:
            result[str(size)] = {
                "n_draws": len(model_tvd),
                "tvd_mean": float(np.mean(model_tvd)),
                "tvd_std": float(np.std(model_tvd)),
                "tvd_p90": float(np.percentile(model_tvd, 90)),
                "sampling_only_tvd_mean": float(np.mean(sampling_tvd)),
            }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=["vader", "tfidf", "transformer"], required=True)
    p.add_argument("--model", default=None, help="model path / HF id (tfidf, transformer)")
    p.add_argument("--name", default=None, help="run name (default: system name)")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=100, help="random batches per game per size")
    p.add_argument("--skip-metacritic", action="store_true")
    p.add_argument("--skip-steam", action="store_true")
    args = p.parse_args()
    if args.system == "transformer" and args.model is None:
        args.model = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    cfg = load_yaml("configs/task1/data.yaml")
    data_dir = get_path("processed") / cfg["output_subdir"]
    clf = build_system(args)
    name = args.name or clf.name
    out_dir = get_path("runs") / "task1" / name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    metrics: dict[str, Any] = {"system": name, "args": vars(args)}

    mc = pd.read_parquet(data_dir / "metacritic.parquet")
    print("[metacritic val] scoring (for the ACC confusion matrix)")
    val = predict(clf, mc[mc.split == "val"], out_dir / "predictions_metacritic_val.parquet")
    rates3 = misclassification_rates(
        confusion_matrix([Polarity(v) for v in val.label], [Polarity(v) for v in val.pred], CLASSES)
    )
    val_bin = val[val.label != "neutral"]
    rates2 = misclassification_rates(
        confusion_matrix(
            [Polarity(v) for v in val_bin.label],
            [Polarity(v) for v in val_bin.pred_binary],
            BINARY_CLASSES,
        )
    )
    metrics["metacritic_val"] = classification_report(
        [Polarity(v) for v in val.label], [Polarity(v) for v in val.pred], CLASSES
    )
    metrics["acc_rates"] = {"three_class": rates3, "binary": rates2}
    # ACC inverts these matrices: a large condition number means the correction is unstable
    # (e.g. a classifier that almost never predicts neutral makes the neutral row a mix of
    # the other two), so ACC can be much worse than plain counting.
    metrics["acc_condition_number"] = {
        "three_class": float(np.linalg.cond(np.array(rates3))),
        "binary": float(np.linalg.cond(np.array(rates2))),
    }

    if not args.skip_metacritic:
        print("[metacritic test] scoring")
        test = predict(clf, mc[mc.split == "test"], out_dir / "predictions_metacritic_test.parquet")
        metrics["metacritic_test"] = {
            "review": classification_report(
                [Polarity(v) for v in test.label], [Polarity(v) for v in test.pred], CLASSES
            ),
            "distribution": distribution_block(test, "game_id", "label", "pred", CLASSES, rates3),
            "batch_ablation": batch_ablation(
                test, "game_id", "label", "pred", CLASSES, args.repeats, rng
            ),
        }

    if not args.skip_steam:
        print("[steam] scoring")
        st = predict(
            clf, pd.read_parquet(data_dir / "steam.parquet"), out_dir / "predictions_steam.parquet"
        )
        metrics["steam"] = {
            "review": classification_report(
                [Polarity(v) for v in st.label],
                [Polarity(v) for v in st.pred_binary],
                BINARY_CLASSES,
            ),
            "predicted_3class_share": st.pred.value_counts(normalize=True).round(4).to_dict(),
            "distribution": distribution_block(
                st, "app_id", "label", "pred_binary", BINARY_CLASSES, rates2
            ),
            "batch_ablation": batch_ablation(
                st, "app_id", "label", "pred_binary", BINARY_CLASSES, args.repeats, rng
            ),
        }

    write_json(out_dir / "metrics.json", metrics)
    print(f"\n=== {name} ===")
    for key in ("metacritic_test", "steam"):
        if key in metrics:
            m = metrics[key]
            cc, acc = m["distribution"]["cc"], m["distribution"].get("acc", {})
            print(
                f"{key:16} acc={m['review']['accuracy']:.3f} macroF1={m['review']['macro_f1']:.3f}"
                f" | MAE={cc['mae']:.3f} TVD={cc['tvd']:.3f} dom={cc['dominant_agreement']:.2f}"
                f" bias={ {k: round(v, 3) for k, v in cc['bias'].items()} }"
                f" | ACC TVD={acc.get('tvd', float('nan')):.3f}"
            )
    print(f"-> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
