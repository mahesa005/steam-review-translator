"""S3/S4: fine-tune BERT-base / RoBERTa-base on Metacritic (3 classes).

- class-weighted cross-entropy (inverse class frequency on train)
- ``--max-length`` is a hyperparameter (try 256 and 512; long reviews get truncated)
- the checkpoint with the best validation macro-F1 is kept; test is not touched here

    pip install -e ".[task1]"
    python experiments/task1_sentiment/train_transformer.py --model roberta-base --max-length 256
    python experiments/task1_sentiment/train_transformer.py --model bert-base-uncased \
        --max-length 512

Output: ``artifacts/models/task1/<run-name>/`` (HF format, loadable by
``TransformerSentimentAnalyzer(model=<dir>)``) plus ``train_log.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from steam_review_translator.core.config import get_path, load_yaml
from steam_review_translator.sentiment.labels import CLASSES
from steam_review_translator.sentiment.transformer import pick_device

LABELS = [c.value for c in CLASSES]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    cfg = load_yaml("configs/task1/data.yaml")
    processed = get_path("processed") / cfg["output_subdir"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--data", type=Path, default=processed / "metacritic.parquet")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--eval-every", type=int, default=2000, help="steps between val evals")
    p.add_argument(
        "--max-train", type=int, default=None, help="subsample train (stratified) for speed"
    )
    p.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="subsample val (stratified); only for smoke tests, never for model selection",
    )
    p.add_argument("--fp16", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def batches(
    df: pd.DataFrame, tok: Any, max_length: int, batch_size: int, shuffle: bool
) -> DataLoader[Any]:
    rows = list(zip(df.text.tolist(), df.label.map(LABEL2ID).tolist(), strict=True))

    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch, strict=True)
        enc = tok(
            list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        )
        enc["labels"] = torch.tensor(labels)
        return dict(enc)

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader[Any], device: str, amp: bool = False) -> float:
    model.eval()
    gold: list[int] = []
    pred: list[int] = []
    for batch in loader:
        labels = batch.pop("labels")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        pred += logits.argmax(-1).cpu().tolist()
        gold += labels.tolist()
    model.train()
    return float(f1_score(gold, pred, average="macro"))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    run_name = args.run_name or f"{args.model.split('/')[-1]}_len{args.max_length}"
    out_dir = get_path("models") / "task1" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    train, val = df[df.split == "train"], df[df.split == "val"]
    if args.max_train and len(train) > args.max_train:
        frac = args.max_train / len(train)
        train = train.groupby("label", group_keys=False).sample(frac=frac, random_state=args.seed)
    if args.max_val and len(val) > args.max_val:
        frac = args.max_val / len(val)
        val = val.groupby("label", group_keys=False).sample(frac=frac, random_state=args.seed)
    print(f"device={device} train={len(train):,} val={len(val):,} run={run_name}")

    counts = train.label.value_counts()
    weights = torch.tensor(
        [len(train) / (len(LABELS) * counts[name]) for name in LABELS], dtype=torch.float
    ).to(device)
    print("class weights:", dict(zip(LABELS, weights.tolist(), strict=True)))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    ).to(device)
    train_loader = batches(train, tok, args.max_length, args.batch_size, shuffle=True)
    val_loader = batches(val, tok, args.max_length, args.batch_size * 2, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total * args.warmup_ratio), total)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    amp = args.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log: dict[str, Any] = {"args": vars(args) | {"data": str(args.data)}, "evals": []}
    best = -1.0
    step = 0
    start = time.time()
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            optim.zero_grad()
            step += 1
            if step % 100 == 0:
                print(
                    f"epoch {epoch} step {step}/{total} loss {loss.item():.4f} "
                    f"({time.time() - start:.0f}s)"
                )
            if step % args.eval_every == 0 or step == total:
                f1 = evaluate(model, val_loader, device, amp)
                log["evals"].append({"step": step, "epoch": epoch, "val_macro_f1": f1})
                print(f"  val macro-F1 {f1:.4f} (best {max(best, f1):.4f})")
                if f1 > best:
                    best = f1
                    model.save_pretrained(out_dir)
                    tok.save_pretrained(out_dir)
                log["best_val_macro_f1"] = best
                (out_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")

    log["best_val_macro_f1"] = best
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"done: best val macro-F1 {best:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
