"""Task 3: meta-review generation (owner: Lukas).

Isi paket:

- ``preprocessing`` : tahap [A] pembersihan & seleksi paragraf.
- ``aspects``       : tahap [B] tabel aspek pseudo untuk training.
- ``serialization`` : tahap [C] format input ``"aspects: ... </s> <teks>"``.
- ``extractive``    : baseline B1 (Lead-k / TextRank).
- ``transformer``   : B2 / E1 / E2 / E3 (BART & T5, zero-shot & fine-tuned).
- ``metrics``       : ROUGE, cakupan aspek, konsistensi polaritas, paired bootstrap.
"""

from steam_review_translator.summarization.aspects import (
    pseudo_aspect_table,
    pseudo_aspect_table_from_reviews,
)
from steam_review_translator.summarization.extractive import (
    ExtractiveMetaReviewGenerator,
    lead_k,
    textrank,
)
from steam_review_translator.summarization.metrics import (
    aspect_coverage,
    average_rouge,
    paired_bootstrap,
    polarity_consistency,
    rouge_scores,
)
from steam_review_translator.summarization.preprocessing import (
    BoilerplateFilter,
    ParagraphSelector,
    PreprocessedDocument,
    preprocess_document,
    preprocess_reviews,
)
from steam_review_translator.summarization.serialization import (
    build_inference_input,
    format_aspect_prefix,
    serialize_input,
)

__all__ = [
    "BoilerplateFilter",
    "ExtractiveMetaReviewGenerator",
    "ParagraphSelector",
    "PreprocessedDocument",
    "aspect_coverage",
    "average_rouge",
    "build_inference_input",
    "format_aspect_prefix",
    "lead_k",
    "paired_bootstrap",
    "polarity_consistency",
    "preprocess_document",
    "preprocess_reviews",
    "pseudo_aspect_table",
    "pseudo_aspect_table_from_reviews",
    "rouge_scores",
    "serialize_input",
    "textrank",
]
