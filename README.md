# MLOps Assignment 2 — Hugging Face fine-tuning, W&B, and Hub

This repository implements the course workflow from *Assignment 2* (IIT Jodhpur PGD AI): modular Python scripts for data preparation, training with the Hugging Face `Trainer`, experiment tracking with [Weights & Biases](https://wandb.ai), final evaluation with an artifact, and optional publishing to the [Hugging Face Hub](https://huggingface.co).

The starter notebook (Google Colab) uses the **UCSD Goodreads** genre setting; these scripts support the same pipeline. For a **fully reproducible quick run** without Drive exports, the default data source is the public **`ag_news`** benchmark (single-label text classification, four classes). To match the assignment narrative in your report, export your Goodreads table from the starter notebook to CSV with columns **`review`** and **`genre`**, then run `data.py` with `--source csv` (see below).

## Project layout

| File | Role |
|------|------|
| `data.py` | Load data, optional stratified subsample, train/validation/test splits, tokenization, write `processed_data/`. |
| `train.py` | Load pre-trained encoder + classification head, `Trainer`, `report_to="wandb"`, optional Hub push. |
| `eval.py` | Test-set metrics, explicit `wandb.log` for final loss/accuracy/F1, `eval_report.json` as a W&B Artifact. |
| `utils.py` | Label map I/O and `compute_metrics` (accuracy + weighted F1). |
| `requirements.txt` | Dependency list for `pip` (matches what the assignment asks you to push to GitHub). |
| `pyproject.toml` | Project metadata and dependencies for **[uv](https://docs.astral.sh/uv/)** (same packages as `requirements.txt`). |
| `uv.lock` | Locked dependency tree produced by `uv lock` (commit to git for reproducible installs). |

## Setup

Use Python **3.10+** (3.11 or 3.12 recommended).

### Option A: uv (recommended if you use Astral uv)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). On Windows, if `uv` is not found after install, add it to `PATH` for the current session or open a new terminal:

```powershell
$env:Path = "C:\Users\helpa\.local\bin;$env:Path"
```

From the project root:

```powershell
cd "c:\Users\helpa\Desktop\mlops assignment 2\mlops_g25ait2148_assignment"
uv lock    # refresh uv.lock from pyproject.toml (after any dependency change)
uv sync    # create/update .venv and install exactly what uv.lock pins
.\.venv\Scripts\Activate.ps1
```

- **`pyproject.toml`** lists direct dependencies; **`uv lock`** resolves the full tree into **`uv.lock`**.
- If **`uv.lock`** looks almost empty, your `pyproject.toml` had no `dependencies` yet — run `uv lock` again after dependencies are filled in.

You can still install from **`requirements.txt`** with uv if you prefer not to use the lockfile:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### Option B: venv + pip (no uv)

```powershell
cd "c:\Users\helpa\Desktop\mlops assignment 2\mlops_g25ait2148_assignment"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### Auth tokens (any setup)

Authenticate (pick what you use):

- **W&B**: `wandb login` (or set `WANDB_API_KEY`).
- **Hugging Face Hub** (for `push_to_hub`): set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` to a write token from [HF settings](https://huggingface.co/settings/tokens).

## How to run (command line)

**1) Prepare data** — default `ag_news` (good for Colab-like GPU runs or local smoke tests):

```powershell
python data.py --source ag_news --sample_per_class 800 --output_dir processed_data
```

Use **your Goodreads CSV** (from the starter notebook) instead:

```powershell
python data.py --source csv --csv_path data\goodreads_reviews.csv --text_col review --label_col genre --output_dir processed_data
```

**2) Train** (creates a W&B run; matches the assignment’s `report_to="wandb"` pattern):

```powershell
python train.py --processed_dir processed_data --output_dir results --wandb_project mlops-assignment2 --wandb_run_name distilbert-run-1
```

Optional: push model + tokenizer after training (public repo id `username/repo-name`):

```powershell
python train.py --hub_model_id YOUR_USERNAME/distilbert-goodreads-genres
```

**3) Evaluate** on the held-out test split, log final metrics, and upload `eval_report.json`:

```powershell
python eval.py --model_dir results --processed_dir processed_data --wandb_project mlops-assignment2 --wandb_run_name distilbert-eval-final
```

For **CPU-only** smoke tests, reduce data and epochs, for example:

```powershell
python data.py --source ag_news --sample_per_class 200 --output_dir processed_data
python train.py --epochs 1 --train_batch_size 8 --eval_batch_size 16 --logging_steps 10
```

## Pre-trained model (assignment Task 3)

Default encoder: **`distilbert-base-uncased`** with `AutoModelForSequenceClassification` and `num_labels` taken from the label map. DistilBERT is smaller and faster than full BERT with only a small accuracy gap, which is why it is a common teaching choice for fine-tuning pipelines.

**For your PDF report (model selection, ~100–150 words):** summarise (1) parameter efficiency vs BERT, (2) suitability for short news/book-style snippets, (3) strong ecosystem support (`Trainer`, Hub, W&B integration), and (4) that your goal is the MLOps workflow rather than chasing leaderboard accuracy.

## Results

Fill this table after you run `eval.py` (values also appear in the W&B run and in `eval_report.json`).

| Metric      | Score |
|------------|-------|
| Accuracy   | _run `eval.py` and paste `eval_accuracy`_ |
| F1 (weighted) | _paste `eval_f1`_ |
| Eval loss  | _paste `eval_loss`_ |

- **Hugging Face model:** `https://huggingface.co/<your-username>/<your-repo>` (after `--hub_model_id` push)
- **W&B project:** `https://wandb.ai/<your-username>/mlops-assignment2`

## Submission checklist (from the assignment brief)

1. Public **GitHub** repo with the Python scripts + **`requirements.txt`** + this README. Including **`pyproject.toml`** and **`uv.lock`** is optional but helps others reproduce the same versions with `uv sync`.
2. Public **Hugging Face** model repo (weights + tokenizer).
3. **W&B** project visible as **public**, with a training dashboard screenshot in the report.
4. **PDF report** (4–5 pages): model choice, training + W&B charts, interpretation of metrics, challenges/learnings — and paste the three links (GitHub, HF, W&B).

## Starter notebook

Course notebook (run once in Colab before refactoring):  
[Starter notebook (Google Colab)](https://colab.research.google.com/drive/15yJsCtRu4kgqCLT44Tjhs3SFOT5GITqC?usp=sharing)
