# Architecture

## Data flow

```
                    PipelineInput {game, reviews[]}
                                 |
               +-----------------+------------------+
               |                                    |
               v                                    v
   Task 1: SentimentAnalyzer           split_sentences(review.text)
   analyze(reviews)                                 |
               |                                    v
               |                       Task 2: AspectTagger
               |                       tag(sentences) -> labels per sentence
               |                                    |
               |                                    v
               |                       aggregate(labels) -> AspectEntry[]
               |                                    |
               v                                    v
     SentimentDistribution               aspect table
               |                                    |
               +-----------------+------------------+
                                 v
          MetaReviewInput {game, reviews, sentiment, aspects}
                                 |
                                 v
                  Task 3: MetaReviewGenerator.generate
                                 |
                                 v
     PipelineOutput {game, sentiment, aspects, meta_review}
```

Tasks 1 and 2 are independent (neither reads the other's output). They run one after the
other today; they could run in parallel later without changing any interface.

## Modules

| Module | Role |
|---|---|
| `core/schemas.py` | pydantic models: the JSON contract between tasks. Extra fields are rejected. |
| `core/interfaces.py` | `typing.Protocol` per task. The pipeline depends only on these. |
| `core/config.py` | YAML loading, project paths (`get_path`), `.env` -> `os.environ`. |
| `core/logging.py` | `get_logger(name)`: stderr + `artifacts/logs/run.log`. |
| `pipeline.py` | `Pipeline.run`, the implementation registries, `build_pipeline(config)`. |
| `cli.py` | `srt demo ...` |
| `summarization/preprocessing.py` | Task 3 step [A]: boilerplate filter, relevance scoring, paragraph selection. |
| `summarization/aspects.py` | Task 3 step [B]: pseudo aspect table for training. |
| `summarization/serialization.py` | Task 3 step [C]: `"aspects: ... </s> <text>"` input format. |
| `summarization/extractive.py` | Task 3 baseline B1: Lead-k and TextRank. |
| `summarization/transformer.py` | Task 3 B2/E1/E2/E3: BART and T5 generators (torch imported lazily). |
| `summarization/metrics.py` | Task 3 evaluation: ROUGE, aspect coverage, polarity consistency, bootstrap. |

## Interfaces

```python
class SentimentAnalyzer(Protocol):
    def analyze(self, reviews: list[Review]) -> SentimentDistribution: ...


class AspectTagger(Protocol):
    def tag(self, sentences: list[str]) -> list[list[AspectLabel]]: ...  # one list per sentence


class MetaReviewGenerator(Protocol):
    def generate(self, inp: MetaReviewInput) -> str: ...
```

Aggregation rule (`aspect/aggregate.py`): `freq` = number of sentences mentioning the
aspect (counted once per sentence); `polarity` = majority polarity, ties -> `neutral`;
sorted by `freq` desc, then aspect name.

## Selecting implementations

`configs/pipeline.yaml` names one implementation per task. `build_pipeline` looks the name
up in that task's registry in `pipeline.py` (name -> `"module:ClassName"`), imports it
lazily, constructs it with `params`, and checks it satisfies the Protocol. Lazy imports keep
torch & co. out of the process unless that implementation is selected.
