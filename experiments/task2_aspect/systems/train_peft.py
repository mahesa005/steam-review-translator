"""Train system C (LoRA aspect model, see ``aspect/peft_tagger.py``) on the train split.

After each epoch the model is scored on val at every threshold in the config; the best
(epoch, threshold) by val pair F1 is kept. Test is never used here.

Usage (from the repo root):
    python experiments/task2_aspect/systems/train_peft.py [--limit N]   # N: train subset
"""

from __future__ import annotations

import json
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from steam_review_translator.aspect.metrics import format_score, score
from steam_review_translator.aspect.peft_tagger import build_model, decode, encode_labels
from steam_review_translator.core.config import load_yaml, project_root
from steam_review_translator.core.schemas import AspectLabel


def load(path: str, split: str) -> tuple[list[str], list[list[AspectLabel]]]:
    rows = [json.loads(x) for x in (project_root() / path).read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == split]
    return [r["sentence"] for r in rows], [[AspectLabel(**x) for x in r["labels"]] for r in rows]


def predict_probs(model, tok, sentences, cfg, device) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(sentences), 64):
            enc = tok(sentences[i : i + 64], truncation=True, max_length=cfg["max_length"],
                      padding=True, return_tensors="pt").to(device)  # fmt: skip
            chunks.append(model(enc["input_ids"], enc["attention_mask"]).softmax(-1).cpu())
    return torch.cat(chunks)


def main() -> int:
    cfg = load_yaml("configs/task2/peft.yaml")
    limit = int(sys.argv[2]) if sys.argv[1:2] == ["--limit"] else None
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(torch.get_num_threads(), 1))

    tr_x, tr_y = load(cfg["labels"], "train")
    va_x, va_y = load(cfg["labels"], "val")
    if limit:
        tr_x, tr_y = tr_x[:limit], tr_y[:limit]
    targets = torch.tensor([encode_labels(y) for y in tr_y])

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = build_model(cfg["base_model"])
    lc = cfg["lora"]
    model.encoder = get_peft_model(
        model.encoder,
        LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
        ),  # fmt: skip
    )
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device {device}; train {len(tr_x)}, val {len(va_x)}; trainable params {n_train:,}")

    t = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    steps = t["epochs"] * ((len(tr_x) + t["batch_size"] - 1) // t["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(steps * t["warmup_ratio"]), steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    out = project_root() / cfg["out_dir"]
    best = {"f1": -1.0}
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        order = list(range(len(tr_x)))
        random.shuffle(order)
        start, total = time.perf_counter(), 0.0
        for k in range(0, len(order), t["batch_size"]):
            idx = order[k : k + t["batch_size"]]
            enc = tok([tr_x[i] for i in idx], truncation=True, max_length=cfg["max_length"],
                      padding=True, return_tensors="pt").to(device)  # fmt: skip
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(enc["input_ids"], enc["attention_mask"]).float()
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]), targets[idx].reshape(-1).to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item() * len(idx)
        probs = predict_probs(model, tok, va_x, cfg, device)
        scores = {th: score(va_y, decode(probs, th)) for th in cfg["thresholds"]}
        th = max(scores, key=lambda x: scores[x]["pair"]["f1"])
        f1 = scores[th]["pair"]["f1"]
        print(f"epoch {epoch}: loss {total / len(tr_x):.4f}, {time.perf_counter() - start:.0f}s; "
              f"val pair F1 {f1:.3f} at threshold {th}")  # fmt: skip
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "threshold": th, "score": scores[th]}
            out.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(out / "adapter")
            torch.save(model.head.state_dict(), out / "head.pt")
            meta = {"base_model": cfg["base_model"], "max_length": cfg["max_length"],
                    "threshold": th, "epoch": epoch, "val_pair_f1": f1,
                    "n_train": len(tr_x), "config": cfg}  # fmt: skip
            (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(format_score(best["score"], f"best: epoch {best['epoch']}, threshold "
                                      f"{best['threshold']} (val)"))  # fmt: skip
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
