"""S1: train TF-IDF + logistic regression on the Metacritic train split.

Hyperparameters (C, n-gram range) are picked on the validation split by macro-F1; the
test split is not touched here.

    python experiments/task1_sentiment/train_tfidf.py
    python experiments/task1_sentiment/train_tfidf.py --out artifacts/models/task1/s1_tfidf.pkl
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from steam_review_translator.core.config import get_path, load_yaml
from steam_review_translator.sentiment.tfidf import TfidfSentimentAnalyzer


def main() -> None:
    cfg = load_yaml("configs/task1/data.yaml")
    processed = get_path("processed") / cfg["output_subdir"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=processed / "metacritic.parquet")
    parser.add_argument("--out", type=Path, default=get_path("models") / "task1" / "s1_tfidf.pkl")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    train, val = df[df.split == "train"], df[df.split == "val"]
    print(f"train={len(train):,} val={len(val):,}")

    best: tuple[float, dict[str, object], Pipeline] | None = None
    for c, ngrams in itertools.product([0.25, 1.0, 4.0], [(1, 1), (1, 2)]):
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=ngrams, min_df=3, max_features=300_000, sublinear_tf=True
                    ),
                ),
                ("lr", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
            ]
        )
        model.fit(train.text, train.label)
        f1 = f1_score(val.label, model.predict(val.text), average="macro")
        params = {"C": c, "ngram_range": list(ngrams)}
        print(f"  {params} val macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, params, model)

    assert best is not None
    f1, params, model = best
    TfidfSentimentAnalyzer(model=model).save(args.out)
    meta = {"val_macro_f1": f1, "params": params}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"best {params} val macro-F1={f1:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
