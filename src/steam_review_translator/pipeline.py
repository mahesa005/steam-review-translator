"""Orchestrates Task 1 -> Task 2 -> Task 3 through the Protocols in ``core.interfaces``."""

from __future__ import annotations

import importlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from steam_review_translator.aspect.aggregate import aggregate
from steam_review_translator.aspect.sentences import split_reviews
from steam_review_translator.core.config import ComponentConfig, PipelineConfig
from steam_review_translator.core.interfaces import (
    AspectTagger,
    MetaReviewGenerator,
    SentimentAnalyzer,
)
from steam_review_translator.core.logging import get_logger
from steam_review_translator.core.schemas import MetaReviewInput, PipelineInput, PipelineOutput

log = get_logger(__name__)

# ---------------------------------------------------------------------------------------
# Implementation registries: name (used in configs/pipeline.yaml) -> "module:ClassName".
# Import paths are resolved lazily, so a real model's heavy deps (torch, bertopic, ...)
# are only imported when that implementation is actually selected.
# Each owner adds their real implementation to their own registry.
# ---------------------------------------------------------------------------------------
SENTIMENT_REGISTRY: dict[str, str] = {
    "mock": "steam_review_translator.sentiment.mock:MockSentimentAnalyzer",
    "vader": "steam_review_translator.sentiment.vader:VaderSentimentAnalyzer",
    "tfidf": "steam_review_translator.sentiment.tfidf:TfidfSentimentAnalyzer",
    "transformer": "steam_review_translator.sentiment.transformer:TransformerSentimentAnalyzer",
}
ASPECT_REGISTRY: dict[str, str] = {
    "mock": "steam_review_translator.aspect.mock:MockAspectTagger",
    "lexicon": "steam_review_translator.aspect.lexicon_tagger:LexiconAspectTagger",
    "peft": "steam_review_translator.aspect.peft_tagger:PeftAspectTagger",
    # "bertopic": "steam_review_translator.aspect.bertopic_tagger:BertopicTagger",  # Mahesa
}
SUMMARIZATION_REGISTRY: dict[str, str] = {
    "mock": "steam_review_translator.summarization.mock:MockMetaReviewGenerator",
    # Lukas: baseline ekstraktif (B1) dan generator seq2seq (B2/E1/E2/E3).
    "extractive": "steam_review_translator.summarization.extractive:ExtractiveMetaReviewGenerator",
    "bart": "steam_review_translator.summarization.transformer:BartMetaReviewGenerator",
    "t5": "steam_review_translator.summarization.transformer:T5MetaReviewGenerator",
}


@contextmanager
def _stage(name: str) -> Iterator[None]:
    start = time.perf_counter()
    log.info("stage %s: start", name)
    yield
    log.info("stage %s: done in %.1f ms", name, (time.perf_counter() - start) * 1000)


class Pipeline:
    """Runs the three tasks on one game's reviews."""

    def __init__(
        self,
        sentiment: SentimentAnalyzer,
        tagger: AspectTagger,
        generator: MetaReviewGenerator,
    ) -> None:
        """Wire the pipeline from any objects satisfying the task Protocols."""
        self.sentiment = sentiment
        self.tagger = tagger
        self.generator = generator

    def run(self, inp: PipelineInput) -> PipelineOutput:
        """Run Task 1 and Task 2 independently, aggregate, then Task 3."""
        log.info("pipeline start: game=%r reviews=%d", inp.game, len(inp.reviews))
        with _stage("task1.sentiment"):
            sentiment = self.sentiment.analyze(inp.reviews)

        with _stage("task2.aspect"):
            per_review = split_reviews([review.text for review in inp.reviews])
            sentences = [s for sents in per_review for s in sents]
            labels = self.tagger.tag(sentences)
            if len(labels) != len(sentences):
                raise ValueError(
                    f"AspectTagger returned {len(labels)} label lists for {len(sentences)} "
                    "sentences; expected one per sentence"
                )
            aspects = aggregate(labels)
            log.info("task2: %d sentences, %d aspects", len(sentences), len(aspects))

        with _stage("task3.meta_review"):
            meta_input = MetaReviewInput(
                game=inp.game, reviews=inp.reviews, sentiment=sentiment, aspects=aspects
            )
            meta_review = self.generator.generate(meta_input)

        return PipelineOutput(
            game=inp.game, sentiment=sentiment, aspects=aspects, meta_review=meta_review
        )


def _import(path: str) -> Any:
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _build(task: str, registry: dict[str, str], cfg: ComponentConfig, protocol: type) -> Any:
    if cfg.impl not in registry:
        raise ValueError(
            f"Unknown {task} implementation {cfg.impl!r}; registered: {sorted(registry)}"
        )
    obj = _import(registry[cfg.impl])(**cfg.params)
    if not isinstance(obj, protocol):
        raise TypeError(f"{task} implementation {cfg.impl!r} does not satisfy {protocol.__name__}")
    log.info("%s: using %r (%s)", task, cfg.impl, type(obj).__name__)
    return obj


def build_pipeline(config: PipelineConfig | None = None) -> Pipeline:
    """Instantiate the implementations named in ``config`` (default: all mocks)."""
    cfg = config or PipelineConfig()
    return Pipeline(
        sentiment=_build("sentiment", SENTIMENT_REGISTRY, cfg.sentiment, SentimentAnalyzer),
        tagger=_build("aspect", ASPECT_REGISTRY, cfg.aspect, AspectTagger),
        generator=_build(
            "summarization", SUMMARIZATION_REGISTRY, cfg.summarization, MetaReviewGenerator
        ),
    )
