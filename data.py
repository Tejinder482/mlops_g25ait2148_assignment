"""Load data, optional balancing sample, train/val/test split, tokenize, save to disk."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from utils import save_label_maps


def _balance_sample(ds, label_key: str, n_per_class: int, seed: int) -> Dataset:
    """Stratified random sample up to n_per_class rows per label (train split only)."""
    import random

    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, row in enumerate(ds):
        lab = int(row[label_key])
        by_label.setdefault(lab, []).append(i)
    keep: list[int] = []
    for _lab, idxs in by_label.items():
        rng.shuffle(idxs)
        keep.extend(idxs[:n_per_class])
    keep.sort()
    return ds.select(keep)


def _load_ag_news() -> tuple[DatasetDict, dict[int, str], dict[str, int]]:
    d = load_dataset("ag_news")
    tv = d["train"].train_test_split(test_size=0.1, seed=42)
    dd = DatasetDict(train=tv["train"], validation=tv["test"], test=d["test"])
    id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    label2id = {v: k for k, v in id2label.items()}
    return dd, id2label, label2id


def _load_csv(
    csv_path: Path, text_col: str, label_col: str, seed: int
) -> tuple[DatasetDict, dict[int, str], dict[str, int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Place your Goodreads export (columns {text_col!r}, "
            f"{label_col!r}) or use --source ag_news for a smoke run."
        )
    raw_tbl = load_dataset("csv", data_files=str(csv_path))["train"]
    if text_col not in raw_tbl.column_names or label_col not in raw_tbl.column_names:
        raise ValueError(
            f"CSV must contain {text_col!r} and {label_col!r}; found {raw_tbl.column_names}"
        )
    uniq = sorted({str(x) for x in raw_tbl[label_col]})
    str_to_id = {g: i for i, g in enumerate(uniq)}
    id2label = {i: g for i, g in enumerate(uniq)}
    label2id = {g: i for i, g in id2label.items()}

    def row_to_std(example):
        return {
            "text": str(example[text_col]),
            "label": str_to_id[str(example[label_col])],
        }

    ds = raw_tbl.map(row_to_std)
    drop = [c for c in ds.column_names if c not in ("text", "label")]
    if drop:
        ds = ds.remove_columns(drop)
    tvt = ds.train_test_split(test_size=0.2, seed=seed)
    val_test = tvt["test"].train_test_split(test_size=0.5, seed=seed + 1)
    dd = DatasetDict(train=tvt["train"], validation=val_test["train"], test=val_test["test"])
    return dd, id2label, label2id

def tokenize_split(
    dd: DatasetDict,
    tokenizer,
    max_length: int,
    text_column: str = "text",
    label_column: str = "label",
) -> DatasetDict:
    def tok(batch):
        enc = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = batch[label_column]
        return enc

    remove = [c for c in dd["train"].column_names if c not in (text_column, label_column)]
    return dd.map(tok, batched=True, remove_columns=remove)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare tokenized datasets for training.")
    p.add_argument(
        "--source",
        choices=["ag_news", "csv"],
        default="ag_news",
        help="ag_news: public text benchmark. csv: your Goodreads (or any) single-label export.",
    )
    p.add_argument("--csv_path", type=Path, default=Path("data/goodreads_reviews.csv"))
    p.add_argument("--text_col", default="review", help="CSV text column name.")
    p.add_argument("--label_col", default="genre", help="CSV label column name.")
    p.add_argument("--sample_per_class", type=int, default=0, help="If >0, subsample train only.")
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--output_dir", type=Path, default=Path("processed_data"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "false")

    if args.source == "ag_news":
        raw, id2label, label2id = _load_ag_news()
    else:
        raw, id2label, label2id = _load_csv(args.csv_path, args.text_col, args.label_col, args.seed)

    if args.sample_per_class and args.sample_per_class > 0:
        raw["train"] = _balance_sample(raw["train"], "label", args.sample_per_class, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tok = tokenize_split(raw, tokenizer, args.max_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_label_maps(args.output_dir / "label_maps.json", id2label, label2id)
    meta = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "source": args.source,
        "csv_path": str(args.csv_path) if args.source == "csv" else None,
        "num_labels": len(id2label),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tok.save_to_disk(str(args.output_dir / "dataset"))
    print(f"Saved tokenized DatasetDict to {args.output_dir / 'dataset'}")
    print(f"Label maps written to {args.output_dir / 'label_maps.json'}")


if __name__ == "__main__":
    main()
