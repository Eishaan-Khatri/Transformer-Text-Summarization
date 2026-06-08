# Benchmark Runbook

This file is the "how to rerun it" note.

The short version: the repo has a small checked model run, a 500-example
baseline run, and a 50-example DistilBART review run. A serious BART-large-CNN
or PEGASUS run should be done on GPU.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Small DistilBART Run

```powershell
python -m src.hf_benchmark --model sshleifer/distilbart-cnn-6-6 --dataset abisee/cnn_dailymail --config 1.0.0 --split test --sample-size 24 --batch-size 2 --max-new-tokens 96
```

This writes:

```text
outputs/metrics/full_benchmark/sshleifer__distilbart-cnn-6-6_summary.json
outputs/metrics/full_benchmark/sshleifer__distilbart-cnn-6-6_examples.csv
```

## Build The Main Tables And Charts

```powershell
python -m src.build_results --run-baselines --sample-size 24
```

This writes the main table, charts, and report:

```text
outputs/tables/benchmark_summary.csv
outputs/figures/
reports/final_report.md
```

## 500-Example Lead Baselines

This run gives a stronger baseline check without overwriting the main report.

```powershell
python -m src.build_results `
  --run-baselines `
  --sample-size 500 `
  --output-dir outputs/metrics/baseline_500 `
  --table-output outputs/tables/baseline_500_summary.csv `
  --figures-output outputs/figures/baseline_500 `
  --report-output reports/baseline_500_report.md
```

## 50-Example DistilBART Review Run

This creates model outputs for manual review and content-discovery files.

```powershell
python -m src.hf_benchmark `
  --model sshleifer/distilbart-cnn-6-6 `
  --dataset abisee/cnn_dailymail `
  --config 1.0.0 `
  --split test `
  --sample-size 50 `
  --batch-size 2 `
  --max-new-tokens 96 `
  --output-dir outputs/metrics/distilbart_50_review
```

Create the review sheet:

```powershell
python -m src.create_error_review_template `
  --examples outputs/metrics/distilbart_50_review/sshleifer__distilbart-cnn-6-6_examples.csv `
  --output outputs/error_analysis/distilbart_50_manual_review_template.csv `
  --model-name sshleifer/distilbart-cnn-6-6 `
  --review-size 50
```

Create tags and similar items:

```powershell
python -m src.content_discovery `
  --examples outputs/metrics/distilbart_50_review/sshleifer__distilbart-cnn-6-6_examples.csv `
  --top-k 5 `
  --output-dir outputs/content_discovery/distilbart_50
```

## Bigger GPU Run

Use this on a CUDA machine:

```powershell
python -m src.run_benchmark_suite `
  --run-baselines `
  --sample-size 500 `
  --batch-size 4 `
  --max-new-tokens 160 `
  --models sshleifer/distilbart-cnn-6-6 facebook/bart-large-cnn google/pegasus-cnn_dailymail `
  --output-dir outputs/metrics/full_benchmark_500 `
  --table-output outputs/tables/full_benchmark_500_summary.csv `
  --figures-output outputs/figures/full_benchmark_500 `
  --report-output reports/full_benchmark_500_report.md
```

The current machine is CPU-only, so I wouldn't treat the current neural-model
numbers as final.
