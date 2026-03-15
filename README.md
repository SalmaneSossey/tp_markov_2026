# TP_Markov2026

Python project for a Markov Chains for Text Generation lab. The project fetches public-domain texts, preprocesses them into a restricted character vocabulary, trains character-level Markov models, evaluates them with log-likelihood and perplexity, and saves generated samples for report writing.

## Setup

### Create a virtual environment

```powershell
python -m venv .venv
```

### Activate it in PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python run_experiments.py
```

## Expected outputs

- Cached raw texts in `data/raw/`
- Preprocessed texts in `data/processed/`
- Metrics JSON files in `outputs/metrics/`
- Generated samples in `outputs/samples/`
- Draft report notes in `report_notes.md`

## Submission checklist

- Project runs from the root with `python run_experiments.py`
- Order-1 and order-3 models produce metrics and samples
- Generated outputs are saved to files
- Report notes are updated with experiment results
