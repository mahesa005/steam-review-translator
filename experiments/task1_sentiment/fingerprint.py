"""Print a short hash of the processed Task 1 splits.

Data prep is re-run on other machines (e.g. Colab); matching fingerprints prove the
splits and labels are identical to the ones used locally.

    python experiments/task1_sentiment/fingerprint.py
"""

from __future__ import annotations

import hashlib

import pandas as pd

from steam_review_translator.core.config import get_path, load_yaml

KEYS = {
    "metacritic.parquet": ["review_id", "game_id", "label", "split", "text"],
    "steam.parquet": ["review_id", "app_id", "label", "text"],
}


def main() -> None:
    cfg = load_yaml("configs/task1/data.yaml")
    data_dir = get_path("processed") / cfg["output_subdir"]
    for name, cols in KEYS.items():
        path = data_dir / name
        if not path.is_file():
            print(f"{name}: missing")
            continue
        df = pd.read_parquet(path, columns=cols).sort_values("review_id")
        h = hashlib.sha256()
        for row in df.itertuples(index=False):
            h.update("\x1f".join(map(str, row)).encode())
            h.update(b"\x1e")
        print(f"{name}: rows={len(df):,} sha256={h.hexdigest()[:16]}")


if __name__ == "__main__":
    main()
