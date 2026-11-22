# Steam Review Translator

IF5153 Advanced NLP, Group 17.

Give it a batch of Steam reviews for **one game** and it returns:

1. **Task 1: sentiment distribution**: share of positive / negative / neutral reviews.
2. **Task 2: aspect table**: which aspects (gameplay, story, performance, ...) players
   mention, how often, and the dominant polarity of each.
3. **Task 3: meta-review**: one paragraph that summarizes the reviews, grounded in the
   aspect table.

> **Status: skeleton.** Every task currently runs a deterministic **mock** (no ML), so the
> pipeline works end to end and the tests pass. Owners replace the mocks with real models.

## Architecture

```
          reviews.json + game name
                     |
              PipelineInput
            /               \
   Task 1 (Farrel)       sentence split -> Task 2 (Mahesa)
   SentimentAnalyzer     AspectTagger -> aggregate()
           |                         |
   SentimentDistribution        AspectEntry[]
            \               /
             MetaReviewInput
                     |
           Task 3 (Lukas) MetaReviewGenerator
                     |
              PipelineOutput
```

Tasks 1 and 2 are independent; Task 3 consumes both plus the original reviews.
Details: [docs/architecture.md](docs/architecture.md).

## Quickstart

Requires Python 3.11+ (3.13 recommended).

> **Windows note:** PyYAML has no wheel for Python 3.15 yet and building it needs MSVC.
> Use Python 3.13 (or 3.12) for the venv: `py -3.13 -m venv .venv`.

```bash
# setup (Linux/macOS, or Git Bash on Windows)
make setup               # .venv + pip install -e ".[dev]" + pre-commit install

# without make
python3.11 -m venv .venv            # Windows: py -3.13 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install

# run the demo on the fake fixture
srt demo --reviews tests/fixtures/sample_reviews.json --game "Fake Game"
python demo.py --reviews tests/fixtures/sample_reviews.json --game "Fake Game"   # same thing
srt demo --reviews tests/fixtures/sample_reviews.json --game "Fake Game" --out artifacts/runs/result.json

# checks
make lint                # ruff check + ruff format --check + mypy
make test                # pytest
make test-cov
```

Copy `.env.example` to `.env` for local settings (`SRT_LOG_LEVEL`, `DATA_DIR`, `LLM_API_KEY`).
Logs go to stderr and `artifacts/logs/run.log`.

## Install extras

The core install is deliberately light (pydantic + pyyaml). Heavy deps are per task:

| Extra | For | Contents |
|---|---|---|
| `.[task1]` | Sentiment | scikit-learn, transformers, torch, vaderSentiment |
| `.[task2]` | Aspects | bertopic, sentence-transformers, umap-learn, hdbscan, spacy, gensim, scikit-learn |
| `.[task3]` | Summarization | transformers, torch, datasets, evaluate, rouge-score, nltk |
| `.[dev]` | Everyone | pytest, pytest-cov, ruff, pre-commit, mypy, types-PyYAML |
| `.[all]` | Everything | all of the above |

e.g. `pip install -e ".[dev,task2]"`. CI installs only `.[dev]`.

## Repo layout

| Path | What |
|---|---|
| `src/steam_review_translator/core/` | Shared contract: schemas, Protocols, config, logging |
| `src/steam_review_translator/sentiment/` | Task 1 |
| `src/steam_review_translator/aspect/` | Task 2 (lexicon, topic mapping, aggregation, sentence split) |
| `src/steam_review_translator/summarization/` | Task 3 (praproses, serialisasi input, baseline ekstraktif, BART/T5, metrik) |
| `src/steam_review_translator/pipeline.py` | Orchestration + implementation registries |
| `src/steam_review_translator/cli.py` | `srt` command |
| `configs/` | `pipeline.yaml` (which impl per task) + per-task configs |
| `experiments/` | Research scripts / notebooks (not part of the library) |
| `data/` | Local datasets, never committed ([data/README.md](data/README.md)) |
| `artifacts/` | Models, embeddings, logs, run outputs, never committed |
| `tests/` | Unit + integration tests, fake fixture |
| `docs/` | Architecture and contributing guides |

## Ownership

| Task | Owner | Code | Experiments | Config |
|---|---|---|---|---|
| 1. Sentiment | Farrel | `sentiment/` | `experiments/task1_sentiment/` | `configs/task1/` |
| 2. Aspects | Mahesa | `aspect/` | `experiments/task2_aspect/` | `configs/task2/` |
| 3. Meta-review | Lukas | `summarization/` | `experiments/task3_summarization/` | `configs/task3/` |
| Shared contract | all | `core/`, `pipeline.py` | | `configs/pipeline.yaml` |

Branch/PR rules: [docs/contributing.md](docs/contributing.md).

## Task 3 (meta-review) quickstart

Task 3 follows the plan in `configs/task3/summarization.yaml`:

- `[A]` `summarization.preprocessing` — drop boilerplate / short / duplicate paragraphs,
  score relevance (TF-IDF vs. game title + aspect lexicon), select paragraphs under a
  token budget (800 for BART, 400 for T5), flag documents under 100 words as invalid.
- `[B]` `summarization.aspects` — pseudo aspect table for training (Task 2 lexicon for
  frequency, Task 1 sentiment model for polarity, top-5 aspects).
- `[C]` `summarization.serialization` — input format
  `"aspects: gameplay (23, positive) ; story (11, positive) </s> <paragraphs>"`.
- `summarization.extractive` — baseline B1 (Lead-k / TextRank).
- `summarization.transformer` — B2 (BART-large-cnn zero-shot), E1 (BART, text only),
  E2 (BART + aspect prefix), E3 (T5 + aspect prefix).
- `summarization.metrics` — ROUGE-1/2/L/Lsum, aspect coverage, polarity consistency,
  paired bootstrap.

```bash
# library only (no ML): extractive baseline inside the pipeline
#   configs/pipeline.yaml -> summarization.impl: extractive
# model runs (needs .[task3])
python experiments/task3_summarization/train_bart.py --output artifacts/models/e2_bart --aspect-prefix
python experiments/task3_summarization/train_t5.py  --output artifacts/models/e3_t5
python experiments/task3_summarization/evaluate.py   --model artifacts/models/e2_bart --name E2 --aspect-prefix
python experiments/task3_summarization/error_analysis.py --system E2=artifacts/runs/e2.json
```

### Mock data Task 1 + Task 2 (untuk Task 3)

`tests/fixtures/mock_task3_input.json` = `MetaReviewInput` lengkap (12 review + distribusi
sentimen Farrel + tabel aspek Mahesa) dan `tests/fixtures/mock_pipeline_output.json` =
`PipelineOutput`. Keduanya dibuat dengan implementasi yang ada di repo, jadi langsung
bisa dipakai sebagai payload model Task 3.

Untuk data sendiri, dua skrip berikut:

```bash
# 1) payload Task 3 dari review apa pun (pakai impl yang dipilih di configs/pipeline.yaml)
python experiments/task3_summarization/make_mock_inputs.py \
    --reviews tests/fixtures/sample_reviews.json --game "Fake Game" --out-dir artifacts/runs/mock

# 2) jalankan model Task 3 terhadap payload itu, lalu review outputnya
python experiments/task3_summarization/run_inference.py \
    --input artifacts/runs/mock/meta_review_input.json --out artifacts/runs/mock/pipeline_output.json

# 3) ubah GameWikiSum lokal (data/raw/Full_GameWiki) jadi dataset training Task 3
python experiments/task3_summarization/build_dataset.py \
    --split train --out data/interim/task3/train.jsonl
# satu DOKUMEN = satu sampel (test: 1.454 dokumen -> 1.449 valid, 5 dibuang <100 kata)
```

GameWikiSum is for academic use only — never commit the dataset or derived files.

## Adding a real implementation

1. **Implement the Protocol** from `core/interfaces.py` in your package, e.g.
   ```python
   # src/steam_review_translator/sentiment/vader.py
   class VaderSentimentAnalyzer:
       def __init__(self, threshold: float = 0.05) -> None: ...
       def analyze(self, reviews: list[Review]) -> SentimentDistribution: ...
   ```
   Import heavy libraries inside that module only.
2. **Register it** in `pipeline.py`:
   ```python
   SENTIMENT_REGISTRY = {
       "mock": "steam_review_translator.sentiment.mock:MockSentimentAnalyzer",
       "vader": "steam_review_translator.sentiment.vader:VaderSentimentAnalyzer",
   }
   ```
3. **Switch it on** in `configs/pipeline.yaml`:
   ```yaml
   sentiment:
     impl: vader
     params: {threshold: 0.05}
   ```
4. Add tests; use `pytest.importorskip("vaderSentiment")` so CI (no heavy extras) skips them.

## Data policy

- **Never commit data** (or model weights, or `.env`). `data/` and `artifacts/` are
  gitignored apart from placeholders.
- `data/raw/` is read-only: derived files go to `data/interim/` or `data/processed/`.
- **GameWikiSum is for academic use only**: do not redistribute it or derived files.
- Sources, licenses, and download steps: [data/README.md](data/README.md).

## JSON contract

Defined in `src/steam_review_translator/core/schemas.py`. Unknown fields are rejected.

**Input** to `srt demo` (a list of reviews):

```json
[
  {"review_id": "r001", "text": "The combat is fantastic. The soundtrack is amazing!", "recommended": true},
  {"review_id": "r002", "text": "Constant crashes and terrible lag.", "recommended": false}
]
```

**`MetaReviewInput`** (what Task 3 receives) and **`PipelineOutput`** (final result):

```json
{
  "game": "Fake Game",
  "reviews": [{"review_id": "r001", "text": "...", "recommended": true}],
  "sentiment": {"positive": 0.5, "negative": 0.3333, "neutral": 0.1667, "dominant": "positive"},
  "aspects": [
    {"aspect": "gameplay", "freq": 4, "polarity": "neutral"},
    {"aspect": "performance", "freq": 4, "polarity": "negative"}
  ]
}
```

```json
{
  "game": "Fake Game",
  "sentiment": {"positive": 0.5, "negative": 0.3333, "neutral": 0.1667, "dominant": "positive"},
  "aspects": [{"aspect": "gameplay", "freq": 4, "polarity": "neutral"}],
  "meta_review": "Across 12 reviews, players' reception of Fake Game is mostly positive ..."
}
```

- `Aspect`: `gameplay, story, graphics, audio, performance, price, multiplayer, difficulty,
  content, overall, other`
- `Polarity`: `positive, negative, neutral`
- `sentiment` shares must sum to 1 (+/- 0.001). `dominant` is computed if omitted (ties -> neutral).
- `aspects` are sorted by `freq` desc. `freq` = number of sentences mentioning the aspect.
