Hand-written interpretation of the tables below; every number cites its CSV under `tables/`.
Anything that depends on the lexicon is **PRELIMINARY** (placeholder seed lexicon).

### (a) Is the corpus usable? **Yes, with cleaning.**

- 6,417,106 reviews over 9,972 games, 0 malformed rows; 95.4% English (fastText, p >= 0.5;
  97.7% among non-short reviews); **4,817,044 clean English reviews (75.1%)**
  (`overview_counts.csv`).
- 2,346 games have >= 200 clean English reviews, so there is plenty to choose from
  (`reviews_per_game_describe.csv`, `candidate_games.csv`). Volume is very skewed: median 57
  reviews per game, top 1% of games hold 37.5% of reviews (`concentration.json`).
- Must-fix noise: **1,024,687 rows (16%) are the literal placeholder "Early Access Review"**
  (`duplicates_top.csv`); they are removed by the short filter (short = 22.3% of all rows,
  `quality_flags.csv`). Spam-like reviews are rare: 1.3% of all rows, 0.9% of English
  (`quality_flags.csv`).
- Not in the data: timestamps (no time analysis possible) and game names (taken from the Steam
  store API for the shortlist). The text still contains **HTML entities** (`&amp;`, `&gt;`:
  see "d amp d", "c amp c" in `nohit_ngrams.csv`) and Steam's ♥♥♥♥ profanity mask; decode
  entities (`html.unescape`) before labeling.
- Labels are skewed: 82.0% of reviews are recommended (`score_distribution.csv`).

### (b) Recommended games and 9/2/2 train/val/test split — PRELIMINARY

Owner decision (2026-11-16): 13 games split **by game** 9 / 2 / 2 (≈ 69 / 15 / 15), so val and
test games are never seen in training. The 13 are the earlier 10 picks plus their 3 backups.

| split | games |
|---|---|
| train (9) | Lara Croft and the Temple of Osiris (289690), HITMAN (236870), Alien Rage - Unlimited (217920), Test Drive Unlimited 2 (9930), Company of Heroes 2 (231430), Transformers: Rise of the Dark Spark (245760), Battlefleet Gothic: Armada (363680), Toren (320820), Tony Hawk's Pro Skater HD (207210) |
| val (2) | Might & Magic: Heroes VI (48220), Paper Sorcerer (263560) |
| test (2) | Sniper: Ghost Warrior 2 (34870), Two Worlds II HD (7520) |
| backup | Grey Goo (290790), Armikrog (334120), Call of Juarez: The Cartel (33420) |

Source and one-line justifications: `../selected_games.csv`. Genres span action-adventure,
shooters, stealth, racing, skateboarding, turn-based and real-time strategy, a puzzle
platformer, an action RPG and a dungeon crawler. All 13 have >= 7 of the 9 aspects at >= 5%
of their lexicon hits (`selected_games_metrics.csv`). Best of 4,290 partitions
(`split_objective.json`, `split_comparison.csv`): train vs val JS divergence 0.0147 bits,
recommend-share gap 0.003, hit-rate gap 0.017; train vs test 0.0102 bits, 0.003, 0.010.
Known imbalance: **multiplayer** is rarer in val (0.056) and test (0.066) than in train
(0.117), and test reviews are shorter (median 205 vs 295 chars in train).

Check before committing to these: the picks lean **negative** (mean recommend share 0.59 vs
0.82 corpus-wide, `split_comparison.csv`, `score_distribution.csv`) because the score rewards
some criticism; they lean towards **action/shooter**; Transformers has only 226 clean reviews.
Swap in a backup if you want a more positive or less action-heavy set.

### (c) Lexicon hit rate (system A) — PRELIMINARY

- **30.6% of sentences hit >= 1 of the 9 aspects** (32.9% of sentences with >= 4 tokens);
  33.0% incl. `overall`; **67.0% of sentences have no label at all**; 62.9% of reviews have
  at least one aspect hit (`lexicon_hit_rate.csv`). 25.3% of sentences have exactly 1 aspect,
  5.3% have >= 2 (`aspects_per_sentence.csv`).
- Hits are concentrated: gameplay 22.4% and story 19.9% of hits; multiplayer 4.5% and
  difficulty 4.9% are thin (`lexicon_hits_per_aspect.csv`).
- Gaps (`nohit_ngrams.csv`, `topic_aspect_purity.csv`): genre/mode words (*turn-based
  strategy, tower defense, open world, single player, fast paced, hack n slash*), *level
  design*, *replay value*, the word *performance* itself, ratings other than `10/10`
  (*8/10*), combat vocabulary (*weapons, enemies, guns*), *voice/dialogue*, *port*, plus
  evaluative sentences with no aspect (*good, fun, great* top the no-hit unigrams).
- Ambiguous keywords (`keyword_quality.csv`): **worth** — 47% of hits are "worth it/the"
  (overall value, not only price); **recommend** — 16% negated ("not recommend"); **sound** —
  10% "sounds like/good"; **balance** — 9% "balance of"; **hours** — 9% playtime statements.
  The five ambiguous words are correctly never matched; their example uses are in the same
  table.

### (d) BERTopic (system B) and what it implies — purity PRELIMINARY

- Defaults (min_cluster_size 10): **543 topics, 44.5% outliers, mean c_v 0.351**. 30: 210
  topics, 45.1%, 0.359. 100: 65 topics, **50.9% outliers, c_v 0.406**. `reduce_outliers`
  (c-TF-IDF) drops outliers to ~0.6% at similar c_v 0.345–0.416 (`topic_runs.csv`).
- Against lexicon labels, size-weighted purity is only 0.54–0.56; 32–55% of topics are mixed
  and 40–57% have < 20% lexicon coverage (`topic_purity_summary.csv`). Clean aspect topics
  exist for price, story, audio, controls, difficulty, multiplayer, puzzles, graphics,
  combat, crashes and fps; but several large topics are not aspects at all: pure sentiment
  ("good, awesome, terrible"), score lines ("10/10, 8/10"), review meta ("negative review,
  update"), and swearing (`topic_aspect_purity.csv`, `mcs100_reduce-none`).
- Implications: **A** is precise on explicit mentions but covers only ~1/3 of sentences, so
  recall is the problem. **B alone** will not work as a tagger: half the sentences are
  outliers, and forcing them into topics (reduce_outliers) just moves the noise. **Hybrid**:
  lexicon for explicit mentions, BERTopic to (1) find missing terms and aspects and (2) label
  sentences in high-purity topics; leave mixed/sentiment-only topics to `overall`/`other` or
  a classifier. Expect the PEFT classifier to be needed for implicit aspects ("each class has
  2 weapons" → gameplay).

### (e) Aspects possibly missing from the 9

From low-coverage topics and themes (`topic_candidate_themes.csv`,
`topic_aspect_purity.csv`): **developers / updates / support** (3,040 sentences in
matching topics), **community / other players** (3,544), **port / platform quality**
(PC vs console/mobile), **modding / Workshop** (387), **atmosphere / horror** and
**nostalgia** (2,448), plus **controls** and **bugs / stability** which form their own clean
topics but sit inside gameplay/performance today. DLC and refunds fit under price; servers fit
under multiplayer. Decide whether to add 1–2 aspects (devs/updates and community are the
strongest) or map them to `other`.

### (f) Manual-verification workload

Owner decision (2026-11-16): an LLM pre-labels every sentence and the owner verifies, with
**75 reviews per game** over the 13 games: 975 reviews → **7,006 sentences** (6,451 with >= 4
tokens). By sentences the split is train 5,154 (74%) / val 1,001 (14%) / test 851 (12%);
Transformers alone has 977 sentences. At 5–10 s per sentence to verify an LLM label that is
**10–19 hours** (`candidate_sample_summary.csv`). For the original 10 games the 75 reviews
are the first 75 of the same seeded per-game shuffle, i.e. a subset of the earlier
150-review draw, so the sample can grow later without relabeling.

### (g) Risks and open questions

1. The lexicon is a placeholder: hit rates, purity, the game ranking and the split will move
   once it is revised. Rerun `make eda` (cached steps make it cheap) after editing it.
2. The selection score is a heuristic (weights in `configs/task2/eda.yaml`); it favours
   criticised games.
3. The sentence splitter treats 2+ spaces as a former line break, and 10.5% of sentences
   come from chunking unpunctuated run-ons (`sentence_summary.csv`). Check splits in the
   labeling sample before labeling.
4. The topic tokenizer strips `/`, so `10/10` becomes `1010`; harmless for the EDA, but
   handle it in system B.
5. Language ID on very short text is unreliable, but those reviews are dropped as short anyway.
6. Embeddings ran on CPU (CUDA torch was not installed; the 2.7 GB download was too slow).
   Results are the same; only the run time differs.
7. Steps 03/04 of this run were produced by the pre-chunking version of `03_lexicon.py`.
   Counts and hit rates match the current code exactly (checked on the smoke sample); the
   `keyword_quality.csv` example sentences and the ambiguous-word counts (taken from a
   400k-sentence sample) differ. Rerun step 03 to regenerate them with the current code.
