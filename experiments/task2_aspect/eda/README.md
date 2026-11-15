# Task 2 EDA (aspect-based opinion mining)

Research code for choosing the labeling corpus and games **before** LLM labeling.
Not part of the library. Results: [`outputs/eda_report.md`](outputs/eda_report.md).
Every change that affects the results is recorded in [`CHANGELOG.md`](CHANGELOG.md).

## Setup (once)

```bash
make setup                      # venv + .[dev]
make eda-setup                  # pip install -e ".[dev,task2]" + spacy download en_core_web_sm
# GPU (optional): the default Windows torch wheel is CPU-only; for CUDA install
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
# (the committed run used CPU torch; only the 50k-sentence embedding step benefits)
```

The raw file must be at `data/raw/steam.csv.bz2` (never modified; step 0 records its
SHA-256 in `outputs/tables/raw_file_fingerprint.json`). Steps 0 and 1 download the fastText
`lid.176.ftz` language model to `artifacts/eda/`; step 4 downloads `all-MiniLM-L6-v2`;
step 5 calls the public Steam store and SteamSpy APIs for ~150 games (cached).

## Run

```bash
make eda-smoke    # every step on the first 50k rows; outputs -> artifacts/eda/smoke_outputs/
make eda          # full run (~1 h on 12 CPU threads); outputs -> outputs/
```

Without `make` (e.g. Windows without Git Bash), run the steps in order from the repo root:

```bash
python experiments/task2_aspect/eda/00_ingest.py           # bz2 CSV -> data/interim parquet (~3 min)
python experiments/task2_aspect/eda/01_overview.py         # language, quality flags, per-game stats
python experiments/task2_aspect/eda/02_sentences.py        # sentence split + lemmas (per-game sample)
python experiments/task2_aspect/eda/03_lexicon.py          # lexicon hit rate, gaps, keyword check
python experiments/task2_aspect/eda/04_topics.py           # BERTopic runs, coherence, purity
python experiments/task2_aspect/eda/05_select_games.py     # score, Steam metadata, 10+3 games, split
python experiments/task2_aspect/eda/06_candidate_sample.py # labeling sample -> data/processed/
python experiments/task2_aspect/eda/07_report.py           # outputs/eda_report.md
```

Flags on every step: `--smoke` (tiny input, separate `*_smoke` files), `--force` (ignore
caches and recompute). Expensive steps are cached: the ingest parquet, review features,
sentences, lexicon hits, embeddings and UMAP projection, Steam metadata.

All parameters (seed 42, sample sizes, filter thresholds, topic settings, selection
weights) are in [`configs/task2/eda.yaml`](../../../configs/task2/eda.yaml).

## Where things go

| What | Where | Committed |
|---|---|---|
| Scripts + `edalib.py` helpers | `experiments/task2_aspect/eda/` | yes |
| Report, figures, tables, `selected_games.csv` | `outputs/` | yes (small) |
| `decisions.md` (hand-written interpretation, embedded in the report) | `outputs/` | yes |
| Parquet intermediates, Steam metadata cache | `data/interim/task2_eda/` | no |
| Labeling sample `candidate_sample.{csv,jsonl}` | `data/processed/` | no |
| fastText model, embeddings, UMAP, smoke outputs | `artifacts/eda/` | no |

## Candidates for promotion to the library

`edalib.split_segments` + `chunk_runon` + `iter_sentences` (sentence splitting that handles
the collapsed line breaks and unpunctuated reviews) and `LemmaLexiconMatcher` (the library
`LexiconMatcher` run on surface + spaCy lemma forms). The library matcher itself already
works (whole-token, multi-word, ambiguous words excluded) and is reused as-is.
