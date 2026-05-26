"""Unified CLI: prepare data, train, evaluate, or run the full pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _invoke(main_fn, script_name: str, argv: list[str]) -> None:
    old = sys.argv
    sys.argv = [script_name, *argv]
    try:
        main_fn()
    finally:
        sys.argv = old


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["ag_news", "csv"], default="ag_news")
    p.add_argument("--csv_path", type=Path, default=Path("data/goodreads_reviews.csv"))
    p.add_argument("--text_col", default="review")
    p.add_argument("--label_col", default="genre")
    p.add_argument("--sample_per_class", type=int, default=0)
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--processed_dir", type=Path, default=Path("processed_data"))
    p.add_argument("--seed", type=int, default=42)


def _add_train_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--hub_model_id", default="")
    p.add_argument("--seed", type=int, default=42)


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model_dir", type=Path, default=Path("results"))
    p.add_argument("--processed_dir", type=Path, default=Path("processed_data"))
    p.add_argument("--wandb_project", default="mlops-assignment2")
    p.add_argument("--wandb_run_name", default="distilbert-eval-final")
    p.add_argument("--report_path", type=Path, default=Path("eval_report.json"))


def _data_argv(args: argparse.Namespace) -> list[str]:
    out = [
        "--source",
        args.source,
        "--csv_path",
        str(args.csv_path),
        "--text_col",
        args.text_col,
        "--label_col",
        args.label_col,
        "--model_name",
        args.model_name,
        "--max_length",
        str(args.max_length),
        "--output_dir",
        str(args.processed_dir),
        "--seed",
        str(args.seed),
    ]
    if args.sample_per_class > 0:
        out.extend(["--sample_per_class", str(args.sample_per_class)])
    return out


def _train_argv(args: argparse.Namespace) -> list[str]:
    out = [
        "--processed_dir",
        str(args.processed_dir),
        "--output_dir",
        str(args.output_dir),
        "--wandb_project",
        args.wandb_project,
        "--wandb_run_name",
        args.wandb_run_name,
        "--epochs",
        str(args.epochs),
        "--train_batch_size",
        str(args.train_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_steps",
        str(args.warmup_steps),
        "--weight_decay",
        str(args.weight_decay),
        "--logging_steps",
        str(args.logging_steps),
        "--seed",
        str(args.seed),
    ]
    if args.hub_model_id.strip():
        out.extend(["--hub_model_id", args.hub_model_id.strip()])
    return out


def _eval_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--model_dir",
        str(args.model_dir),
        "--processed_dir",
        str(args.processed_dir),
        "--wandb_project",
        args.wandb_project,
        "--wandb_run_name",
        args.wandb_run_name,
        "--report_path",
        str(args.report_path),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLOps assignment pipeline (data -> train -> eval).",
        epilog="Examples:\n"
        "  python main.py prepare --source ag_news --sample_per_class 800\n"
        "  python main.py train --epochs 3\n"
        "  python main.py eval\n"
        "  python main.py all --source ag_news --sample_per_class 200 --epochs 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Prepare tokenized data (data.py).")
    _add_data_args(prep)

    train_p = sub.add_parser("train", help="Fine-tune classifier (train.py).")
    _add_train_args(train_p)

    eval_p = sub.add_parser("eval", help="Evaluate on test split (eval.py).")
    _add_eval_args(eval_p)

    all_p = sub.add_parser("all", help="Run prepare -> train -> eval in sequence.")
    _add_data_args(all_p)
    all_p.add_argument("--output_dir", type=Path, default=Path("results"))
    all_p.add_argument("--wandb_project", default="mlops-assignment2")
    all_p.add_argument("--wandb_run_name", default="distilbert-run-1")
    all_p.add_argument("--epochs", type=int, default=3)
    all_p.add_argument("--train_batch_size", type=int, default=16)
    all_p.add_argument("--eval_batch_size", type=int, default=32)
    all_p.add_argument("--learning_rate", type=float, default=3e-5)
    all_p.add_argument("--warmup_steps", type=int, default=100)
    all_p.add_argument("--weight_decay", type=float, default=0.01)
    all_p.add_argument("--logging_steps", type=int, default=50)
    all_p.add_argument("--hub_model_id", default="")
    all_p.add_argument("--report_path", type=Path, default=Path("eval_report.json"))
    all_p.add_argument(
        "--eval_run_name",
        default="distilbert-eval-final",
        help="W&B run name for the final eval step.",
    )
    all_p.add_argument("--skip_prepare", action="store_true")
    all_p.add_argument("--skip_train", action="store_true")
    all_p.add_argument("--skip_eval", action="store_true")

    args = parser.parse_args()

    from data import main as data_main
    from eval import main as eval_main
    from train import main as train_main

    if args.command == "prepare":
        _invoke(data_main, "data.py", _data_argv(args))
    elif args.command == "train":
        _invoke(train_main, "train.py", _train_argv(args))
    elif args.command == "eval":
        _invoke(eval_main, "eval.py", _eval_argv(args))
    elif args.command == "all":
        processed = args.processed_dir
        dataset_dir = processed / "dataset"
        if not args.skip_prepare:
            _invoke(data_main, "data.py", _data_argv(args))
        elif not (processed / "meta.json").is_file() or not dataset_dir.is_dir():
            parser.error(
                f"Missing processed data under {processed}. "
                "Run without --skip_prepare or: python main.py prepare ..."
            )
        if not args.skip_train:
            _invoke(train_main, "train.py", _train_argv(args))
        elif not args.output_dir.is_dir():
            parser.error(f"Missing model at {args.output_dir}. Run train first or omit --skip_train.")
        if not args.skip_eval:
            eval_ns = argparse.Namespace(
                model_dir=args.output_dir,
                processed_dir=args.processed_dir,
                wandb_project=args.wandb_project,
                wandb_run_name=args.eval_run_name,
                report_path=args.report_path,
            )
            _invoke(eval_main, "eval.py", _eval_argv(eval_ns))


if __name__ == "__main__":
    main()
