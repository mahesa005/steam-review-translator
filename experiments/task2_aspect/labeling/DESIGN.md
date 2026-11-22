# Task 2 labeling pipeline: design

Status: **approved by the Task 2 owner on 2026-11-16** (all **[proposed]** items accepted,
including the open questions in section 10 as proposed). Implementation in progress.

## 1. Goal

Produce trustworthy sentence-level aspect labels for the Task 2 labeling sample, with as little
manual work as possible: an LLM labels, three independent LLM reviewers check, the main LLM makes
the final call, and the owner verifies only what is uncertain.

- **Input:** `data/processed/candidate_sample.jsonl` (7,002 sentences, 975 reviews, 13 games;
  train 5,150 / val 1,001 / test 851).
- **Output:** `data/processed/task2_labels.jsonl`, one record per sentence with the final labels,
  plus a verification queue for the owner. Labels match the library contract:
  a list of `AspectLabel(aspect, polarity)` per sentence, `[]` when there is no aspect.

## 2. Label set **[owner: keep 9 aspects]**

| Label | Meaning |
|---|---|
| `gameplay`, `story`, `graphics`, `audio`, `performance`, `price`, `multiplayer`, `difficulty`, `content` | the 9 aspects |
| `overall` | a verdict on the whole game, not on one aspect ("10/10", "highly recommend", "don't buy") |
| `other` | an aspect-like topic outside the 9 (developers/updates/support, community/other players, modding, ports/platform) |
| `[]` (empty list) | the sentence mentions no aspect and gives no overall verdict |

Polarity (`positive` / `negative` / `neutral`) is the opinion **towards that aspect**, not the
tone of the whole sentence. One sentence can carry several labels with different polarities.
Each aspect appears at most once per sentence.

**[proposed] Decision rules for known ambiguities** (from the EDA keyword check):

| Case | Rule |
|---|---|
| "worth the money / price", "not worth $20" | `price` |
| "worth it", "worth playing", "worth your time" (no money mentioned) | `overall` |
| "(not) recommend", "10/10", "8/10", "buy it" | `overall`, polarity follows negation |
| "sounds like / sounds good" | not `audio` unless the sound itself is discussed |
| "hours played", "200 hours in" (playtime statement only) | `[]`; `content` only when length/amount of content is evaluated |
| implicit mentions ("each class has 2 weapons") | label the aspect (`gameplay`), polarity `neutral` if no opinion |
| sarcasm | label the intended meaning |
| devs/updates, community, mods, ports | `other` |

These rules become the labeling guideline (section 5). They are tuned on the pilot (train data
only) and frozen before the full run.

## 3. Pipeline

```
candidate_sample.jsonl
   │  make_batches: per game, whole reviews kept together, ~50 sentences per batch (~140 batches)
   ▼
Stage 1  LABEL     Opus 5.5 (effort medium)      → labels for every sentence in the batch
   ▼
Stage 2  REVIEW    3 × Sonnet 5.5 (effort low)   → each reviewer, independently, per sentence:
                                                    OK | FLAG (issue, reasoning, suggested labels)
                                                    + one batch-level verdict
   ▼
Stage 3  FINALIZE  Opus 5.5 (effort medium)      → for every sentence flagged by >= 1 reviewer:
                                                    final labels + kept/changed + rationale.
                                                    Unflagged sentences keep their stage-1 labels.
   ▼
Stage 4  VERIFY    owner                         → all flagged or changed sentences
                                                    + 10% random of unflagged ones (per split)
   ▼
task2_labels.jsonl  (final = owner-verified where checked, stage-3 labels otherwise)
```

**Why each choice:**
- **Whole reviews as context [owner]:** a sentence like "It runs terribly" is only clear with the
  review around it. The LLM sees the full review but labels only the listed sentences.
- **Batches of ~50 sentences:** large enough for caching and consistent labeling, small enough
  for the reviewers to check every item carefully. A review is never split across batches.
- **3 independent reviewers:** each sees the guideline, the reviews and the stage-1 labels, but
  not the other reviewers' output, so their flags are independent.
- **Stage 3 only on flagged sentences [proposed]:** an unflagged sentence was accepted by all three
  reviewers; re-deciding it costs money without new information.
- **Verification sample [owner]:** the random 10% of unflagged sentences measures how often all
  three reviewers miss an error. That rate is reported with the dataset.

## 4. Data formats

All model outputs are constrained with structured outputs (JSON schema), so every response parses.

- **Stage 1 output (per sentence):** `{"sentence_id", "labels": [{"aspect", "polarity"}]}`
- **Stage 2 output (per reviewer):**
  `{"items": [{"sentence_id", "verdict": "ok"|"flag", "issue"?, "reasoning"?, "suggested_labels"?}],
  "batch_verdict": "pass"|"needs_revision", "notes"}`
  where `issue` ∈ `wrong_aspect`, `missing_aspect`, `extra_aspect`, `wrong_polarity`, `guideline`.
- **Stage 3 output (per flagged sentence):**
  `{"sentence_id", "final_labels", "decision": "kept"|"changed", "rationale"}`
- **Final record:** sentence fields from the sample + `labels`, `source`
  (`stage1` | `stage3` | `owner`), `flag_count`, `changed`, `verified`.

## 5. Prompts

Stored as versioned files in `experiments/task2_aspect/labeling/prompts/` (committed); each run
records the SHA-256 of the prompts it used.

- `guideline.md`: label definitions, decision rules (section 2), 15–20 worked examples from
  train games only. Shared by all stages and placed first in the system prompt so it is cached.
- `label.md`, `review.md`, `finalize.md`: the per-stage task instructions.

## 6. Running and storage

- **Message Batches API [owner]:** each stage is submitted as one batch job (50% cheaper,
  asynchronous, typically done within hours, at most 24 h). Stage 2 starts after stage 1 ends;
  stage 3 after stage 2.
- **Prompt caching:** the guideline is identical in every request, so it is cached.
- **Thinking/effort:** Opus 5.5 always thinks; effort is set explicitly to `medium`. Sonnet 5.5
  runs with effort `low`.
- **Failures:** requests that error, expire, or end with `stop_reason: "refusal"` are collected
  and re-submitted (refusal fallbacks are not available on the Batches API).
- **Storage (not committed, it is data):**
  `data/interim/task2_labeling/<run_id>/` holds the requests, raw results, and parsed outputs
  per stage; `data/processed/task2_labels.jsonl` holds the final labels. Every raw response is
  kept, because these models cannot be run with a fixed temperature, so a rerun may differ.
- **API key:** read from `ANTHROPIC_API_KEY` in `.env` (`.env.example` will be updated).

## 7. Pilot first [owner]

1. Run all stages on **2 batches from train games** (~100 sentences).
2. Owner checks every pilot label; measure flag rate, change rate, cost per batch.
3. Fix the guideline and prompts; rerun the pilot if the changes are large.
4. Freeze the prompts (hash recorded in `../eda/CHANGELOG.md`-style log), then run the rest.

Val and test sentences are never used to tune the guideline.

## 8. Cost (rough estimate, to be measured in the pilot)

Assumptions: ~140 batches; guideline ~3k tokens (cached); ~2k tokens of reviews per batch;
~2k output tokens per stage-1 call; reviewers ~1.5k output each; stage 3 on ~30% of sentences.
At Batch API prices this is roughly **$25–60 for the full run**. The pilot reports the real
per-batch cost before the full run is started.

## 9. Not in scope

- Building the Task 2 taggers (systems A, B, hybrid, PEFT); they consume these labels.
- Labeling more than the 75-review-per-game sample.

## 10. Decisions (2026-11-16)

1. Decision rules in section 2: approved.
2. Stage 3 runs only on flagged sentences: approved.
3. A sentence goes to stage 3 when **>= 1 of 3** reviewers flags it: approved.
4. All stages split sentences with the library splitter
   (`steam_review_translator.aspect.sentences`), the same one the pipeline uses.
