"""Fine-tune a sequence classifier with Hugging Face Trainer and Weights & Biases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import compute_metrics_builder, load_label_maps


def _training_args(
    output_dir: Path,
    epochs: int,
    train_bs: int,
    eval_bs: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    logging_steps: int,
    run_name: str,
    seed: int,
) -> TrainingArguments:
    """Build TrainingArguments compatible across common transformers versions."""
    common = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="wandb",
        run_name=run_name,
        seed=seed,
        fp16=torch.cuda.is_available(),
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def main() -> None:
    p = argparse.ArgumentParser(description="Train DistilBERT (or compatible) text classifier.")
    p.add_argument("--processed_dir", type=Path, default=Path("processed_data"))
    p.add_argument("--output_dir", type=Path, default=Path("results"))
    p.add_argument("--wandb_project", default="mlops-assignment2")
    p.add_argument("--wandb_run_name", default="distilbert-run-1")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--hub_model_id", default="", help="If set (user/repo), push model+tokenizer after training.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta_path = args.processed_dir / "meta.json"
    maps_path = args.processed_dir / "label_maps.json"
    data_path = args.processed_dir / "dataset"
    if not meta_path.is_file() or not maps_path.is_file() or not data_path.is_dir():
        raise FileNotFoundError(
            f"Missing processed data under {args.processed_dir}. Run: python data.py ..."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_name = meta["model_name"]
    max_length = int(meta["max_length"])
    id2label, label2id = load_label_maps(maps_path)
    num_labels = len(id2label)

    ds = load_from_disk(str(data_path))

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model": model_name,
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "max_length": max_length,
            "dataset": meta.get("source", "unknown"),
            "num_labels": num_labels,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    targs = _training_args(
        args.output_dir,
        args.epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.learning_rate,
        args.warmup_steps,
        args.weight_decay,
        args.logging_steps,
        args.wandb_run_name,
        args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if args.hub_model_id.strip():
        try:
            from huggingface_hub import login

            tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if tok:
                login(token=tok)
            trainer.model.push_to_hub(args.hub_model_id.strip())
            tokenizer.push_to_hub(args.hub_model_id.strip())
            url = f"https://huggingface.co/{args.hub_model_id.strip()}"
            wandb.run.summary["huggingface_model"] = url
            print(f"Pushed to Hugging Face Hub: {url}")
        except Exception as e:
            print(f"Hub push skipped or failed: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
