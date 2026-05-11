"""Shared helpers: label maps, tokenized dataset prep, and Trainer metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction


def label_maps_from_names(names: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """Build id2label / label2id with stable sorted order."""
    unique = sorted(set(names))
    id2label = {i: unique[i] for i in range(len(unique))}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def save_label_maps(path: Path, id2label: dict[int, str], label2id: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": label2id,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_label_maps(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in raw["id2label"].items()}
    label2id = dict(raw["label2id"])
    return id2label, label2id


def compute_metrics_builder() -> Callable[[EvalPrediction], dict[str, float]]:
    """Return compute_metrics(pred) for Hugging Face Trainer (accuracy + weighted F1)."""

    def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
        labels = pred.label_ids
        logits = pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    return compute_metrics
