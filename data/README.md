# Data

**Nothing in this folder is committed except this README and the `.gitkeep` placeholders.**
Every teammate downloads the data locally. Point `DATA_DIR` (in `.env`) elsewhere if you
keep data outside the repo.

## Folders

| Folder       | Meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| `raw/`       | Original downloads, exactly as obtained. **Never modify or overwrite.** |
| `interim/`   | Cleaned / filtered / split intermediates. Reproducible from `raw/`.     |
| `processed/` | Final model-ready datasets (train/dev/test). Reproducible from `raw/`.  |

Anything derived must be reproducible from `raw/` by a script in `experiments/` or the
package. Record the script that produced each file.

## Datasets

> Placeholders: fill in the exact source URL, version/date, size, and license before use.

| Dataset | Used by | Source | License / terms | Local path |
|---|---|---|---|---|
| Steam Review Dataset 2017 (`steam.csv.bz2`, 6.4M English reviews, no header: `app_id, review_text, review_score (1/-1), review_votes`) | Task 1, Task 2 | [Zenodo 1000885](https://zenodo.org/records/1000885) | CC BY-NC 4.0 | `raw/steam/steam.csv.bz2` |
| Metacritic's Best Games and Reviews 2025 (`games.csv`, `games_reviews.csv`: 1.13M user + 0.52M critic reviews) | Task 1 | [Kaggle davutb/metacritic-games](https://www.kaggle.com/datasets/davutb/metacritic-games) | _check Kaggle page_ (scraped from Metacritic) | `raw/metacritic/metacritic-games.zip` |
| Labelled aspect sentences (if any) | Task 2 | _TODO_ | _TODO_ | `raw/aspect/` |
| GameWikiSum | Task 3 | _TODO: URL from the paper_ | **Academic / research use only.** Do not redistribute. | `raw/gamewikisum/` |

## How to download

```bash
make task1-download     # Metacritic zip + Steam bz2 (no Kaggle login needed)
make task1-data         # -> data/processed/task1/ (see configs/task1/data.yaml)
# TODO (other tasks): replace with the real commands, e.g.
# kaggle datasets download -d <owner>/<dataset> -p data/raw
# curl -L -o data/raw/gamewikisum.zip <URL>
```

## Rules

- Never commit data, even small samples. Test fixtures with *fake* reviews live in
  `tests/fixtures/`.
- Never edit files in `raw/` in place: write outputs to `interim/` or `processed/`.
- Respect each dataset's license. GameWikiSum is academic use only.
