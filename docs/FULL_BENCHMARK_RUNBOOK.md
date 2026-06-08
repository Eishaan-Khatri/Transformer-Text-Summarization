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

## Larger Run

For a stronger result, run the same commands with a larger sample size. I would
not call the current numbers final because 24 examples is small.
