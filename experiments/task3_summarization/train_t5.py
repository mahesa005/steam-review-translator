"""Fine-tuning T5 untuk Task 3 (owner: Lukas) - skenario E3.

Sama persis dengan ``train_bart.py`` (E1/E2), hanya model dan learning rate berbeda:
T5-base, lr 1e-4, max input 512 token, budget paragraf 400 token.

E3 dibandingkan dengan E2 untuk menjawab RQ3: BART atau T5 lebih cocok untuk tugas ini.

Contoh:
    python experiments/task3_summarization/train_t5.py \
        --output artifacts/models/e3_t5 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  # pastikan paket repo bisa diimpor tanpa pip install -e
from common import (
    DEFAULT_DATASET,
    build_examples,
    load_split,
    load_task3_config,
    make_compute_metrics,
    normalize_rows,
    resolve_fields,
)

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.logging import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Argumen CLI skenario E3."""
    parser = argparse.ArgumentParser(description="Fine-tune T5 for Task 3 (meta-review)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/task3/summarization.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--no-aspect-prefix", action="store_true", help="E1-like: teks saja")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--budget-tokens", type=int, default=None)
    parser.add_argument("--pilot-samples", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--max-val-samples", type=int, default=200, help="val docs for ROUGE-L model selection"
    )
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    """Jalankan fine-tuning T5 dan simpan model + tokenizer."""
    args = parse_args()
    cfg = load_task3_config(args.config)
    train_cfg = cfg.get("training", {})
    pre_cfg = cfg.get("preprocessing", {})
    asp_cfg = cfg.get("aspects", {})
    dec_cfg = cfg.get("decoding", {})
    inp_cfg = cfg.get("input", {})

    epochs = args.epochs if args.epochs is not None else train_cfg.get("num_train_epochs", 3)
    lr = args.lr if args.lr is not None else train_cfg.get("learning_rate_t5", 1e-4)
    batch_size = args.batch_size or train_cfg.get("per_device_batch_size", 4)
    grad_accum = args.grad_accum or train_cfg.get("gradient_accumulation_steps", 8)
    max_input = args.max_input_tokens or inp_cfg.get("max_input_tokens", 512)
    max_target = args.max_target_tokens or inp_cfg.get("max_new_tokens", 256)
    budget = args.budget_tokens or pre_cfg.get("budget_tokens", 400)
    top_k = asp_cfg.get("top_k", 5)
    num_beams = args.num_beams or dec_cfg.get("num_beams", 4)
    length_penalty = args.length_penalty or dec_cfg.get("length_penalty", 1.0)

    import torch  # import lokal
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    matcher = LexiconMatcher(load_lexicon())

    train_rows = load_split(args.train_split, dataset_name=args.dataset)
    val_rows = load_split(args.val_split, dataset_name=args.dataset)
    para_key, sum_key, title_key = resolve_fields(list(train_rows[0].keys()))
    train_rows = normalize_rows(train_rows, para_key, sum_key, title_key)
    val_rows = normalize_rows(val_rows, para_key, sum_key, title_key)

    # --pilot-samples 0 = full train split; not given = config default.
    pilot = (
        args.pilot_samples
        if args.pilot_samples is not None
        else train_cfg.get("pilot_train_samples")
    )
    if pilot:
        train_rows = train_rows[:pilot]
        val_rows = val_rows[: max(pilot // 10, 1)]
    # ROUGE-L on val needs generation every epoch: cap val so it stays affordable.
    val_rows = val_rows[: args.max_val_samples]

    train_examples, train_stats = build_examples(
        train_rows,
        tokenizer=tokenizer,
        matcher=matcher,
        budget_tokens=budget,
        max_target_tokens=max_target,
        top_k=top_k,
        use_aspect_prefix=not args.no_aspect_prefix,
    )
    val_examples, val_stats = build_examples(
        val_rows,
        tokenizer=tokenizer,
        matcher=matcher,
        budget_tokens=budget,
        max_target_tokens=max_target,
        top_k=top_k,
        use_aspect_prefix=not args.no_aspect_prefix,
    )
    log.info("train stats %s | val stats %s", train_stats, val_stats)

    def preprocess_fn(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        """Tokenize batch untuk T5 (target diprefix "summarize: " agar instruksi jelas)."""
        inputs = tokenizer(
            batch["input"],
            max_length=max_input,
            truncation=True,
        )
        labels = tokenizer(
            text_target=["summarize: " + t for t in batch["target"]],
            max_length=max_target,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = Dataset.from_dict(
        {
            "input": [e.input_text for e in train_examples],
            "target": [e.target for e in train_examples],
        }
    )
    val_ds = Dataset.from_dict(
        {
            "input": [e.input_text for e in val_examples],
            "target": [e.target for e in val_examples],
        }
    )
    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=["input", "target"])
    val_ds = val_ds.map(preprocess_fn, batched=True, remove_columns=["input", "target"])

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    # length_penalty dipakai saat beam search saat evaluasi (spec decoding).
    model.generation_config.length_penalty = length_penalty
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        fp16=args.fp16 or bool(train_cfg.get("fp16", False)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_num_beams=num_beams,
        generation_max_length=max_target,
        seed=args.seed,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    log.info("model disimpan ke %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
