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
| Steam reviews dump (`steam.csv.bz2`) | Task 1, Task 2 | _TODO: URL (e.g. Kaggle Steam reviews)_ | _TODO: check dataset page_ | `raw/steam.csv.bz2` |
| Labelled aspect sentences (if any) | Task 2 | _TODO_ | _TODO_ | `raw/aspect/` |
| GameWikiSum | Task 3 | _TODO: URL from the paper_ | **Academic / research use only.** Do not redistribute. | `raw/gamewikisum/` |

## How to download

```bash
# TODO: replace with the real commands, e.g.
# kaggle datasets download -d <owner>/<dataset> -p data/raw
# curl -L -o data/raw/gamewikisum.zip <URL>
```

## Rules

- Never commit data, even small samples. Test fixtures with *fake* reviews live in
  `tests/fixtures/`.
- Never edit files in `raw/` in place: write outputs to `interim/` or `processed/`.
- Respect each dataset's license. GameWikiSum is academic use only.
