# Task 2 systems: results and limitations

Gold labels: `data/processed/task2_labels.jsonl` (7,002 sentences labeled by Claude Opus; see
`../labeling/DESIGN.md`). Train 5,150 / val 1,001 / test 851 sentences, split by game.
Model choices (lexicon thresholds, epoch, decision threshold) were made on val; each system was
scored on test once. Scores: `artifacts/task2/eval/<impl>_<split>.json` (from `evaluate.py`).

## Test results

| System | Pair P | Pair R | Pair F1 | Aspect F1 | Aspect macro F1 | Polarity acc | Exact |
|---|---|---|---|---|---|---|---|
| Mock (seed lexicon, keyword polarity) | 0.289 | 0.110 | 0.159 | 0.410 | 0.473 | 0.388 | 0.209 |
| A: induced lexicon + VADER | 0.314 | 0.316 | 0.315 | 0.545 | 0.552 | 0.578 | 0.219 |
| **C: roberta-base + LoRA** | **0.613** | **0.561** | **0.586** | **0.732** | **0.714** | **0.801** | **0.474** |

- Pair = an (aspect, polarity) pair counts only if both match. Aspect = detection only.
- C per aspect (F1): graphics 0.90, story 0.83, price 0.82, audio 0.80, overall 0.76,
  gameplay 0.73, performance 0.71, multiplayer 0.67, content 0.66, difficulty 0.64, other 0.35.
- C training: 10 epochs (~30 s each on an RTX 4050), best epoch 9, threshold 0.45,
  val pair F1 0.621; val F1 plateaued from epoch 6 (0.619).

## Limitations

1. **Sentence-only input.** The tagger sees one sentence, while the LLM labeler read the whole
   review. Sentences whose aspect or polarity depends on context ("Never again.", "It's worth
   every penny" after a price complaint) cannot be resolved. This is likely the largest source
   of error.
2. **Silver labels.** Gold labels come from an LLM, not people. Only the test split and part of
   val/train went through a second-model review (which changed 0.4% of labels), and the owner's
   manual check (`data/processed/task2_verify_queue.csv`) was not done before these results.
   Scores measure agreement with Opus, and label noise caps them.
3. **Weak aspects.** `other` (recall 0.23) is a heterogeneous catch-all; `difficulty` and
   `content` overlap with `gameplay` and have few examples. `multiplayer` has only 4 val
   sentences, so its val-based choices are unreliable.
4. **One global threshold** on 1 - P(none) for all aspects; rare aspects would likely prefer
   their own.
5. **No search over models or hyperparameters.** One base model (roberta-base), one LoRA setup
   (r 16 on query/value), one learning rate; one seed, so no variance estimate.
6. **System A polarity** is one sentence-level VADER score copied to every aspect in the
   sentence, so mixed sentences ("great story but terrible controls") get one polarity.
7. **Domain and size.** 13 games, English reviews only, 5,150 training sentences.
8. System B (BERTopic mapping) and a hybrid were planned but not built.

## Possible next steps

Neighbouring sentences or the review as context; per-aspect thresholds tuned on val; LoRA on
all linear layers or a stronger encoder (deberta-v3-base, roberta-large); class weights for rare
aspects; several seeds; finishing the manual verification and re-scoring on verified test labels.
