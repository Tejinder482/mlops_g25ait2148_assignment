# MLOps Assignment 2 — Fine-tuning + W&B + Hugging Face Hub

This repo is my submission for *MLOps Assignment 2 (IIT Jodhpur PGD AI)*. It takes the starter Colab notebook workflow and turns it into normal Python scripts you can run from the terminal:

- prepare data
- train a Hugging Face model and track it in W&B
- run final evaluation and save a report
- (optional but required for marks) push the trained model to the Hugging Face Hub

The starter notebook uses the **UCSD Goodreads** genre dataset. If you don’t have the Goodreads CSV handy, you can still test the full pipeline using the public **`ag_news`** dataset (4 classes). That’s what `--source ag_news` is for.

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

Use Python **3.12**. (Python 3.13 often causes install issues for `torch`/`transformers` on Windows.)

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

## How to run (copy/paste)

### Quick test run (recommended first)

This is the easiest way to confirm “my code works”.

**Step 1: Prepare data**

```powershell
python main.py prepare --source ag_news --sample_per_class 200 --processed_dir processed_data
```

**Step 2: Train (small run)**

```powershell
wandb login
python main.py train --processed_dir processed_data --output_dir results --epochs 1 --train_batch_size 8 --eval_batch_size 16 --logging_steps 10 --wandb_project mlops-assignment2 --wandb_run_name smoke-run
```

**Step 3: Evaluate**

```powershell
python main.py eval --model_dir results --processed_dir processed_data --wandb_project mlops-assignment2 --wandb_run_name smoke-eval
```

After Step 3, you will see Accuracy / F1 / Loss in the terminal and you will also get an `eval_report.json` file.

### Using your Goodreads CSV (from the Colab notebook)

If you exported your Goodreads data as a CSV with columns `review` and `genre`:

```powershell
python data.py --source csv --csv_path data\goodreads_reviews.csv --text_col review --label_col genre --output_dir processed_data
python train.py --processed_dir processed_data --output_dir results --wandb_project mlops-assignment2 --wandb_run_name distilbert-run-1
python eval.py --model_dir results --processed_dir processed_data --wandb_project mlops-assignment2 --wandb_run_name distilbert-eval-final
```

### Push model to Hugging Face (Task 6)

After training finishes, push the model + tokenizer to your Hugging Face profile:

```powershell
$env:HF_TOKEN="your_huggingface_write_token"
python train.py --processed_dir processed_data --output_dir results --hub_model_id YOUR_USERNAME/distilbert-goodreads-genres
```

## Pre-trained model (assignment Task 3)

Default encoder: **`distilbert-base-uncased`** with `AutoModelForSequenceClassification` and `num_labels` taken from the label map. DistilBERT is smaller and faster than full BERT with only a small accuracy gap, which is why it is a common teaching choice for fine-tuning pipelines.

**For your PDF report (model selection, ~100–150 words):** summarise (1) parameter efficiency vs BERT, (2) suitability for short news/book-style snippets, (3) strong ecosystem support (`Trainer`, Hub, W&B integration), and (4) that your goal is the MLOps workflow rather than chasing leaderboard accuracy.

## Results

Fill this after you run `eval.py` (the numbers will print in the terminal).

| Metric      | Score |
|------------|-------|
| Accuracy   | 0.87145 |
| F1 (weighted) | 0.86951 |
| Eval loss  | 0.46378 |

- **Hugging Face model:** `https://huggingface.co/tejinder482/distilbert-agnews-smoke` (replace after push)
- **W&B project:** `https://wandb.ai/tejindersingh2202-indian-institute-of-technology/mlops-assignment2`

## Submission checklist (from the assignment brief)

1. Public **GitHub** repo with the Python scripts + **`requirements.txt`** + this README. Including **`pyproject.toml`** and **`uv.lock`** is optional but helps others reproduce the same versions with `uv sync`.
2. Public **Hugging Face** model repo (weights + tokenizer).
3. **W&B** project visible as **public**, with a training dashboard screenshot in the report.
4. **PDF report** (4–5 pages): model choice, training + W&B charts, interpretation of metrics, challenges/learnings — and paste the three links (GitHub, HF, W&B).

## Starter notebook

Course notebook (run once in Colab before refactoring):
[Starter notebook (Google Colab)](https://colab.research.google.com/drive/15yJsCtRu4kgqCLT44Tjhs3SFOT5GITqC?usp=sharing)
