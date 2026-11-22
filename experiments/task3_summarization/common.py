"""Helper bersama untuk eksperimen Task 3 (owner: Lukas).

Dipakai oleh ``train_bart.py``, ``train_t5.py``, ``evaluate.py``, dan
``error_analysis.py``.

Data: GameWikiSum (split resmi 11.744 train / 1.454 val / 1.454 test).
Lisensi GameWikiSum hanya untuk riset akademik (lihat README & data/README.md).

Tahapan yang diimplementasikan di sini (spec bagian 3):
  [A] pembersihan & seleksi paragraf  -> ``preprocess_document``
  [B] tabel aspek pseudo untuk training -> ``pseudo_aspect_table``
  [C] serialisasi input                 -> ``serialize_input``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  # pastikan paket repo bisa diimpor tanpa pip install -e

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.config import load_yaml
from steam_review_translator.core.logging import get_logger
from steam_review_translator.summarization.aspects import (
    lexicon_polarity_fn,
    pseudo_aspect_table,
)
from steam_review_translator.summarization.preprocessing import preprocess_document
from steam_review_translator.summarization.serialization import (
    format_aspect_prefix,
    serialize_input,
)

log = get_logger(__name__)

#: Nama dataset GameWikiSum di HuggingFace Hub (opsional, bisa diganti lewat CLI).
DEFAULT_DATASET = "Yueh-Han/GameWikiSum"

#: Nama field kandidat pada dataset (dicek otomatis saat dimuat).
FIELD_PARAGRAPHS = ("preprocessed_paragraphs_per_reviews", "paragraphs", "raw_reviews")
FIELD_SUMMARY = ("preprocessed_summary", "summary")
FIELD_TITLE = ("game_title", "game_name", "title")


@dataclass
class Example:
    """Satu sampel training/evaluasi Task 3."""

    index: int
    game: str
    input_text: str
    target: str
    aspects: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def load_task3_config(path: str | Path = "configs/task3/summarization.yaml") -> dict[str, Any]:
    """Baca YAML konfigurasi Task 3."""
    return load_yaml(path)


def resolve_fields(columns: list[str]) -> tuple[str, str, str]:
    """Pilih nama field paragraf, ringkasan, dan judul game dari kolom dataset.

    Mengembalikan tuple ``(paragraf, ringkasan, judul)``. Melempar ``ValueError``
    dengan daftar kolom yang tersedia bila field wajib tidak ditemukan.
    """
    lowered = {c.lower(): c for c in columns}

    def pick(candidates: tuple[str, ...]) -> str | None:
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        return None

    paragraphs = pick(FIELD_PARAGRAPHS)
    summary = pick(FIELD_SUMMARY)
    title = pick(FIELD_TITLE)
    missing = [
        name
        for name, value in (("paragraphs", paragraphs), ("summary", summary), ("title", title))
        if value is None
    ]
    if missing:
        raise ValueError(
            f"GameWikiSum: field {missing} tidak ditemukan; kolom yang ada: {sorted(columns)}"
        )
    return paragraphs or "", summary or "", title or ""


def load_split(
    split: str,
    *,
    dataset_name: str = DEFAULT_DATASET,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Muat satu split GameWikiSum sebagai list of dict.

    ``dataset_name`` boleh berupa folder lokal berisi ``train.json`` / ``val.json`` /
    ``test.json`` (file asli GameWikiSum, lihat build_dataset.py); selain itu dimuat dari
    Hugging Face Hub (butuh extra ``.[task3]``, package ``datasets``).
    """
    if Path(dataset_name).is_dir():
        return load_local_split(Path(dataset_name), split)

    from datasets import load_dataset  # import lokal: library berat

    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    log.info("loaded %s: %d rows", split, len(ds))
    return [dict(row) for row in ds]


#: Nama split -> file pada folder GameWikiSum lokal.
LOCAL_SPLIT_FILES = {"train": "train.json", "validation": "val.json", "val": "val.json",
                     "test": "test.json"}  # fmt: skip
#: Batas paragraf per dokumen saat memuat file lokal (beberapa dokumen punya ribuan).
MAX_LOCAL_PARAGRAPHS = 400


def load_local_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Muat satu split dari file GameWikiSum asli (array kelompok -> array dokumen).

    Hanya field yang dipakai yang disimpan (hemat memori): ``game_title``,
    ``preprocessed_paragraphs_per_reviews`` (diratakan jadi list paragraf, maksimal
    ``MAX_LOCAL_PARAGRAPHS``) dan ``preprocessed_summary``.
    """
    path = data_dir / LOCAL_SPLIT_FILES.get(split, f"{split}.json")
    rows: list[dict[str, Any]] = []
    for group in iter_json_array(path):
        for doc in group if isinstance(group, list) else [group]:
            paragraphs: list[str] = []
            for review in doc.get("preprocessed_paragraphs_per_reviews") or []:
                paragraphs.extend([review] if isinstance(review, str) else review)
                if len(paragraphs) >= MAX_LOCAL_PARAGRAPHS:
                    break
            rows.append(
                {
                    "game_title": str(doc.get("game_title") or doc.get("wiki_title") or ""),
                    "preprocessed_paragraphs_per_reviews": paragraphs[:MAX_LOCAL_PARAGRAPHS],
                    "preprocessed_summary": doc.get("preprocessed_summary")
                    or doc.get("summary")
                    or "",
                }
            )
    log.info("loaded %s from %s: %d documents", split, path, len(rows))
    return rows


def normalize_rows(
    rows: list[dict[str, Any]],
    para_key: str,
    sum_key: str,
    title_key: str,
) -> list[dict[str, Any]]:
    """Samakan nama field antar baris dataset jadi ``paragraphs``/``summary``/``title``."""
    return [
        {**row, "paragraphs": row[para_key], "summary": row[sum_key], "title": row[title_key]}
        for row in rows
    ]


def truncate_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    """Potong teks ke ``max_tokens`` token pertama (spec: target 256 token)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_examples(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    matcher: LexiconMatcher | None = None,
    budget_tokens: int = 800,
    max_target_tokens: int = 256,
    top_k: int = 5,
    use_aspect_prefix: bool = True,
    polarity_fn: Any = None,
) -> tuple[list[Example], dict[str, float]]:
    """Bangun list :class:`Example` dari baris GameWikiSum (tahap [A] + [B] + [C]).

    Mengembalikan ``(examples, stats)``; ``stats`` berisi persentase sampel valid
    (>= 100 kata setelah pembersihan) sesuai spec [A] langkah 5.
    """
    matcher = matcher or LexiconMatcher(load_lexicon())
    polarity_fn = polarity_fn or lexicon_polarity_fn
    examples: list[Example] = []
    kept = 0
    for i, row in enumerate(rows):
        paragraphs = row.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        summary = row.get("summary") or ""
        game = str(row.get("title") or "unknown game")
        doc = preprocess_document(
            game,
            list(paragraphs),
            matcher=matcher,
            budget_tokens=budget_tokens,
            token_counter=lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"]),
        )
        if not doc.is_valid:
            continue  # spec [A] langkah 5: buang sampel < 100 kata
        kept += 1
        target = truncate_to_tokens(summary, tokenizer, max_target_tokens)
        table = pseudo_aspect_table(doc.text, matcher, polarity_fn, top_k=top_k)
        prefix = format_aspect_prefix(table, top_k=top_k)
        input_text = serialize_input(prefix, doc.text) if use_aspect_prefix else doc.text
        examples.append(
            Example(
                index=i,
                game=game,
                input_text=input_text,
                target=target,
                aspects=[e.model_dump() for e in table],
                meta={
                    "words": doc.words,
                    "tokens": doc.tokens,
                    "paragraphs_kept": len(doc.paragraphs),
                    "paragraphs_total": doc.total_paragraphs,
                    "dropped_boilerplate": doc.dropped_boilerplate,
                    "dropped_short": doc.dropped_short,
                    "dropped_duplicate": doc.dropped_duplicate,
                },
            )
        )
    stats = {
        "total": float(len(rows)),
        "valid": float(kept),
        "valid_ratio": (kept / len(rows)) if rows else 0.0,
    }
    log.info("built %d/%d valid examples (%.1f%%)", kept, len(rows), 100 * stats["valid_ratio"])
    return examples, stats


def iter_json_array(path: str | Path, *, chunk_size: int = 1_048_576) -> Any:
    """Yield elemen-elemen array JSON teratas satu per satu (streaming).

    Dipakai untuk file GameWikiSum yang ukurannya sangat besar (train.json ~2 GB):
    seluruh file tidak pernah dimuat ke memori sekaligus.
    """
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as fh:
        buf = fh.read(chunk_size)
        start = buf.find("[")
        if start == -1:
            raise ValueError(f"{path}: bukan array JSON")
        idx = start + 1
        while True:
            # Lewati whitespace dan koma pemisah elemen.
            while True:
                while idx < len(buf) and buf[idx] in " \t\r\n,":
                    idx += 1
                if idx < len(buf):
                    break
                more = fh.read(chunk_size)
                if not more:
                    return
                buf = more
                idx = 0
            if buf[idx] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, idx)
            except json.JSONDecodeError:
                # Elemen belum lengkap -> baca lebih banyak data.
                more = fh.read(chunk_size)
                if not more:
                    raise
                buf = buf[idx:] + more
                idx = 0
                continue
            yield obj
            idx = end
            # Buang bagian buffer yang sudah dipakai supaya memori tetap kecil.
            if idx > chunk_size:
                buf = buf[idx:]
                idx = 0


def generate_outputs(
    model_name: str,
    examples: list[Example],
    tokenizer: Any,
    max_input_tokens: int = 1024,
    max_target_tokens: int = 256,
    *,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    batch_size: int = 8,
    device: str | None = None,
) -> list[str]:
    """Generate keluaran model untuk list :class:`Example` (dipakai B2 & evaluasi).

    Memakai beam search sesuai spec bagian 4.
    """
    import torch  # import lokal: library berat
    from transformers import AutoModelForSeq2SeqLM  # import lokal

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    outputs: list[str] = []
    texts = [e.input_text for e in examples]
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_target_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def save_jsonl(items: list[dict[str, Any]], path: str | Path) -> Path:
    """Simpan list of dict sebagai JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Baca JSONL menjadi list of dict."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_compute_metrics(tokenizer: Any) -> Any:
    """``compute_metrics`` for ``Seq2SeqTrainer`` (``predict_with_generate=True``).

    Returns ROUGE-1/2/L/Lsum on the validation set, so ``metric_for_best_model="rougeL"``
    and early stopping work.
    """
    import numpy as np

    from steam_review_translator.summarization.metrics import average_rouge

    def compute(eval_pred: Any) -> dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return average_rouge(decoded_preds, decoded_labels)

    return compute
