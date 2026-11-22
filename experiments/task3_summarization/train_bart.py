"""Fine-tuning BART untuk Task 3 (owner: Lukas).

Skenario:
  E1 : BART-base fine-tuned, input teks saja           (``--no-aspect-prefix``)
  E2 : BART-base fine-tuned + prefix aspek             (default)
  B2 : tanpa fine-tuning (zero-shot)                    (``--zero-shot``)

Hyperparameter awal (spec bagian 4): lr 3e-5, batch efektif 32 (4 x grad accum 8),
epoch 3-5 dengan early stopping pada ROUGE-L validation, fp16, max input/target
1.024/256 token, beam 4, no_repeat_ngram_size 3.

Contoh:
    python experiments/task3_summarization/train_bart.py \
        --output artifacts/models/e2_bart --aspect-prefix --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  # pastikan paket repo bisa diimpor tanpa pip install -e
from common import (
    DEFAULT_DATASET,
    build_examples,
    generate_outputs,
    load_split,
    load_task3_config,
    normalize_rows,
    resolve_fields,
)

from steam_review_translator.aspect.lexicon import LexiconMatcher, load_lexicon
from steam_review_translator.core.logging import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Argumen CLI skenario E1/E2/B2."""
    parser = argparse.ArgumentParser(description="Fine-tune BART for Task 3 (meta-review)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="nama dataset GameWikiSum")
    parser.add_argument("--model", default="facebook/bart-base", help="model HF")
    parser.add_argument("--output", type=Path, required=True, help="direktori output model")
    parser.add_argument("--config", default="configs/task3/summarization.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--aspect-prefix", action="store_true", help="E2: pakai prefix aspek")
    parser.add_argument("--zero-shot", action="store_true", help="B2: tanpa fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="default dari config")
    parser.add_argument("--lr", type=float, default=None, help="default dari config")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--budget-tokens", type=int, default=None)
    parser.add_argument("--pilot-samples", type=int, default=None, help="subset untuk uji cepat")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    """Jalankan training (atau evaluasi zero-shot) dan simpan model."""
    args = parse_args()
    cfg = load_task3_config(args.config)
    train_cfg = cfg.get("training", {})
    pre_cfg = cfg.get("preprocessing", {})
    asp_cfg = cfg.get("aspects", {})
    dec_cfg = cfg.get("decoding", {})
    inp_cfg = cfg.get("input", {})

    epochs = args.epochs if args.epochs is not None else train_cfg.get("num_train_epochs", 3)
    lr = args.lr if args.lr is not None else train_cfg.get("learning_rate_bart", 3e-5)
    batch_size = args.batch_size or train_cfg.get("per_device_batch_size", 4)
    grad_accum = args.grad_accum or train_cfg.get("gradient_accumulation_steps", 8)
    max_input = args.max_input_tokens or inp_cfg.get("max_input_tokens", 1024)
    max_target = args.max_target_tokens or inp_cfg.get("max_new_tokens", 256)
    budget = args.budget_tokens or pre_cfg.get("budget_tokens", 800)
    top_k = asp_cfg.get("top_k", 5)
    num_beams = args.num_beams or dec_cfg.get("num_beams", 4)
    length_penalty = args.length_penalty or dec_cfg.get("length_penalty", 1.0)

    import torch  # import lokal: library berat
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

    # ------------------------------------------------------------- data
    train_rows = load_split(args.train_split, dataset_name=args.dataset)
    val_rows = load_split(args.val_split, dataset_name=args.dataset)
    para_key, sum_key, title_key = resolve_fields(list(train_rows[0].keys()))
    train_rows = normalize_rows(train_rows, para_key, sum_key, title_key)
    val_rows = normalize_rows(val_rows, para_key, sum_key, title_key)

    pilot = args.pilot_samples or train_cfg.get("pilot_train_samples")
    if pilot:
        train_rows = train_rows[:pilot]
        val_rows = val_rows[: max(pilot // 10, 1)]

    train_examples, train_stats = build_examples(
        train_rows,
        tokenizer=tokenizer,
        matcher=matcher,
        budget_tokens=budget,
        max_target_tokens=max_target,
        top_k=top_k,
        use_aspect_prefix=args.aspect_prefix,
    )
    val_examples, val_stats = build_examples(
        val_rows,
        tokenizer=tokenizer,
        matcher=matcher,
        budget_tokens=budget,
        max_target_tokens=max_target,
        top_k=top_k,
        use_aspect_prefix=args.aspect_prefix,
    )
    log.info("train stats %s | val stats %s", train_stats, val_stats)

    # ------------------------------------------------------------- zero-shot (B2)
    if args.zero_shot:
        outputs = generate_outputs(args.model, train_examples, tokenizer, max_input, max_target)
        print(f"zero-shot sample:\n{outputs[0] if outputs else '(tidak ada data)'}")
        return 0

    # ------------------------------------------------------------- training (E1/E2)
    def preprocess_fn(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        """Tokenize batch: input dipotong ke max_input, target ke max_target."""
        inputs = tokenizer(
            batch["input"],
            max_length=max_input,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    from datasets import Dataset

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
    # Tokenisasi sekali di awal (lebih cepat daripada preprocess_function per batch).
    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=["input", "target"])
    val_ds = val_ds.map(preprocess_fn, batched=True, remove_columns=["input", "target"])

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    # length_penalty dipakai saat beam search saat evaluasi (spec decoding).
    model.generation_config.length_penalty = length_penalty
    args_out = Seq2SeqTrainingArguments(
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
        args=args_out,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    log.info("model disimpan ke %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
