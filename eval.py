"""Final test evaluation, W&B metric logging, and classification-report artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wandb
from datasets import load_from_disk
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

from utils import compute_metrics_builder, load_label_maps


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate saved checkpoint on the test split.")
    p.add_argument("--model_dir", type=Path, default=Path("results"))
    p.add_argument("--processed_dir", type=Path, default=Path("processed_data"))
    p.add_argument("--wandb_project", default="mlops-assignment2")
    p.add_argument("--wandb_run_name", default="distilbert-eval-final")
    p.add_argument("--report_path", type=Path, default=Path("eval_report.json"))
    args = p.parse_args()

    maps_path = args.processed_dir / "label_maps.json"
    data_path = args.processed_dir / "dataset"
    meta_path = args.processed_dir / "meta.json"
    id2label, _label2id = load_label_maps(maps_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    test_ds = load_from_disk(str(data_path))["test"]

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={"phase": "final_eval", "model_dir": str(args.model_dir), **meta},
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(),
    )

    eval_results = trainer.evaluate(test_ds)
    print(eval_results)

    wandb.log(
        {
            "final/loss": float(eval_results.get("eval_loss", float("nan"))),
            "final/accuracy": float(eval_results.get("eval_accuracy", float("nan"))),
            "final/f1": float(eval_results.get("eval_f1", float("nan"))),
        }
    )

    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    labels = list(test_ds["labels"])

    names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        labels,
        preds,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(str(args.report_path))
    wandb.log_artifact(artifact)

    wandb.finish()
    print(f"Wrote {args.report_path} and uploaded W&B artifact 'eval-report'.")


if __name__ == "__main__":
    main()
