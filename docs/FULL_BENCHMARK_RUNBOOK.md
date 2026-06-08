# Full Benchmark Runbook

## Purpose

This runbook explains how to reproduce real CNN/DailyMail benchmark metrics.
Do not publish metric claims until this run has completed and outputs are saved.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Small Smoke Run

```powershell
python -m src.hf_benchmark --model facebook/bart-large-cnn --split test --sample-size 16 --batch-size 2
```

This checks dependency, model-download, tokenization, generation, and ROUGE
plumbing.

## Portfolio-Usable Run

```powershell
python -m src.hf_benchmark --model facebook/bart-large-cnn --split test --sample-size 1024 --batch-size 16
python -m src.hf_benchmark --model google/pegasus-cnn_dailymail --split test --sample-size 1024 --batch-size 8
```

Record the hardware, date, sample size, model name, batch size, and generated
JSON files.

## Stronger Run

Only run this if hardware and time allow:

```powershell
python -m src.hf_benchmark --model facebook/bart-large-cnn --split test --sample-size 11490 --batch-size 32
```

This uses the full CNN/DailyMail test split and can be slow.

## Result Acceptance Rules

- Metrics must come from generated output files.
- Compression ratio must be reported with its formula.
- Batch size must be shown with hardware details.
- Do not round upward.
- If only a subset is evaluated, say subset size clearly.

