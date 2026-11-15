# Task 2 EDA change log

Every change to the EDA's data, parameters or code that affects results, newest first.
Each entry says what changed, why, and what it did to the outputs, so numbers in
`outputs/eda_report.md` can be traced back to the run that produced them.

## 2026-11-16

### One sentence splitter for the pipeline, the EDA and the labeling sample
- **What:** the splitter (HTML decoding, line-break/2+-space segments, spaCy sentencizer, run-on
  chunking) moved from `edalib.py` to the library, `steam_review_translator.aspect.sentences`.
  The pipeline now uses it (`split_reviews`) instead of the old regex splitter; `edalib`
  re-exports it. spaCy (no model) became a core dependency (owner decision).
- **Why:** the labeling sample and the pipeline split reviews differently, so labels made on the
  sample would not match the sentences a tagger sees in the pipeline.
- **Impact:** none on any EDA output: the labeling sample is byte-identical, and the pipeline
  splitter reproduces all 7,002 sample sentences exactly. `spacy.blank("en")` gives the same
  boundaries as `en_core_web_sm` (checked on all 975 sample reviews) and is ~20x faster.

### Step 03 rerun with the current (chunked) code
- **What:** reran `03_lexicon.py` on the cached lexicon hits.
- **Why:** the committed step 03 tables came from the first full run, made with the older,
  non-chunked version of the script.
- **Impact:** only `tables/keyword_quality.csv` changed. All hit rates, per-aspect counts,
  keyword frequencies, per-game stats and no-hit n-grams are byte-identical. Ambiguous-word
  counts are now over all 2,685,750 sentences instead of a 400k-sentence sample (e.g. *hard*
  32,524 sentences, was 4,835), and the five example sentences per keyword were re-drawn
  (reservoir sampling, seed 42). Suspicious shares of all regular keywords are unchanged.

### HTML entities decoded in the labeling sample
- **What:** `06_candidate_sample.py` decodes HTML entities (`&amp;`, `&gt;`, `&lt;`, and
  double-encoded ones such as `&amp;amp;trade;`) before splitting into sentences
  (`edalib.unescape_html`, with a unit test).
- **Why:** 54 of 7,006 sample sentences contained raw entities, which would confuse the LLM
  pre-labeling and manual verification.
- **Impact:** same 975 reviews; 7,006 → **7,002 sentences**. In 36 reviews, 61 sentence texts
  changed and 4 sentence ids disappeared (the `;` inside `&gt;` had been creating extra
  sentence breaks). No new ids. Steps 01–04 still use the raw text; entities affect well under
  1% of their sentences.

### 13 games, 9/2/2 train/val/test split, 75 reviews per game
- **What:** `selection.n_games` 10 → 13 (`n_train` 9, `n_val` 2, `n_test` 2);
  `labeling_sample.reviews_per_game` 50 → 75. `05_select_games.py` searches all 4,290
  train/val/test partitions.
- **Why:** owner decision: add a validation set at about 70/15/15. With 10 games a by-game
  split cannot hit 70/15/15, so the 3 backups were promoted (9/2/2 ≈ 69/15/15) and 3 new
  backups were picked. 75 reviews per game because an LLM will pre-label and the owner only
  verifies.
- **Impact:** train 9 / val 2 / test 2 games; 7,006 sentences (before the HTML fix above),
  about 10–19 h of verification. Multiplayer is rarer in val and test than in train. For the
  original 10 games the 75 reviews are a subset of the earlier 150-review draw.
- **Commits:** `2d6c261`, `22fa7c4`.

### Labeling sample reduced to 50 reviews per game (superseded the same day)
- **What:** `labeling_sample.reviews_per_game` 150 → 50.
- **Why:** 10,145 sentences (14–28 h) was too much to verify.
- **Impact:** 3,609 sentences. Replaced by the 75-review decision above.
- **Commit:** `5f42620`.

## 2026-11-14

### Committed outputs: LF newlines, no trailing whitespace
- **What:** report and JSON summaries are written with LF and one final newline; truncated
  example snippets are stripped.
- **Why:** the pre-commit hooks rewrote them otherwise.
- **Impact:** formatting only. **Commit:** `af54bb9`.

### `game_stats.csv` limited to games with >= 200 reviews
- **What:** the committed per-game table keeps 2,744 of 9,972 games.
- **Why:** the full table (663 KB) exceeded the 500 KB pre-commit limit.
- **Impact:** every candidate game is still included; the all-games distribution is in
  `reviews_per_game_describe.csv`.

### Memory-safe steps 03 and 04
- **What:** step 03 matches and aggregates in 250k-sentence chunks; step 04 samples sentence
  ids before loading any text.
- **Why:** the background run was stopped twice by the system for low memory (15 GB RAM,
  about 2 GB free with other apps open).
- **Impact:** identical numbers on the smoke sample (checked); much lower peak memory.

### CPU-only torch
- **What:** installed the CPU build of torch instead of CUDA.
- **Why:** the 2.7 GB CUDA wheel was downloading at about 420 kB/s.
- **Impact:** only run time (embedding 50,000 sentences took 150 s on CPU). Results unchanged.

### Schema of the raw file confirmed
- **What:** `data/raw/steam.csv.bz2` has no header; columns confirmed by the owner as
  `app_id, review_text, review_score (1/-1), review_votes (helpful flag, metadata only)`.
- **Why:** required before any analysis; there are no timestamp or game-name columns.
- **Impact:** no time analysis; names and genres come from the Steam store API for the
  shortlisted games.

### First full run
- Steps 00–06 on all 6,417,106 reviews. Raw file sha256 `35f4d0c7…2c4da4`, unchanged
  afterwards (checked against the value recorded before the run).
