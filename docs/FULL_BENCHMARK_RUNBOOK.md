# Benchmark Runbook

These are the commands I used for the checked result.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run DistilBART

```powershell
python -m src.hf_benchmark --model sshleifer/distilbart-cnn-6-6 --dataset abisee/cnn_dailymail --config 1.0.0 --split test --sample-size 24 --batch-size 2 --max-new-tokens 96
```

This writes:

```text
outputs/metrics/full_benchmark/sshleifer__distilbart-cnn-6-6_summary.json
outputs/metrics/full_benchmark/sshleifer__distilbart-cnn-6-6_examples.csv
```

## Build Baselines, Tables, and Charts

```powershell
python -m src.build_results --run-baselines --sample-size 24
```

This writes:

```text
outputs/tables/benchmark_summary.csv
outputs/figures/rouge_comparison.png
outputs/figures/compression_ratio.png
outputs/figures/throughput.png
outputs/figures/*.svg
reports/final_report.md
```

## 500-Example Lead Baseline Run

This command writes a larger baseline-only run without overwriting the main
24-example report:

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

This command creates the model outputs used for manual error review and content
discovery:

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

Then create the review sheet:

```powershell
python -m src.create_error_review_template `
  --examples outputs/metrics/distilbart_50_review/sshleifer__distilbart-cnn-6-6_examples.csv `
  --output outputs/error_analysis/distilbart_50_manual_review_template.csv `
  --model-name sshleifer/distilbart-cnn-6-6 `
  --review-size 50
```

And create the summary-tag / neighbor retrieval files:

```powershell
python -m src.content_discovery `
  --examples outputs/metrics/distilbart_50_review/sshleifer__distilbart-cnn-6-6_examples.csv `
  --top-k 5 `
  --output-dir outputs/content_discovery/distilbart_50
```

## Larger Run

For a stronger result, use `src.run_benchmark_suite` on a CUDA machine:

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

The current machine is CPU-only, so I would not treat the current neural-model
numbers as a final benchmark.
