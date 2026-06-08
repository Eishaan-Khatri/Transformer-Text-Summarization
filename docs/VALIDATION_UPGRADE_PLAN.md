# Validation Upgrade Plan

This project is useful, but it still has a ceiling.

Right now it shows that I can build the workflow. It does not yet prove that I
ran a serious large-model benchmark.

The reason is simple: this machine is CPU-only. DistilBART already takes about
14 seconds per example here. BART-large-CNN and PEGASUS would be slower.

## What Is Already Done

| Check | Status | File |
|---|---|---|
| Lead baselines on 500 examples | Done | `outputs/tables/baseline_500_summary.csv` |
| DistilBART on 50 examples | Done | `outputs/metrics/distilbart_50_review/` |
| Latency per example | Done | summary JSON files and CSV tables |
| Manual review sheet | Ready | `outputs/error_analysis/distilbart_50_manual_review_template.csv` |
| Summary tags and neighbors | Done | `outputs/content_discovery/distilbart_50/` |
| Labelled retrieval score | Not done | no labels in the current rows |
| BART-large-CNN on 500 examples | Not done | needs GPU or a long CPU run |
| PEGASUS on 500 examples | Not done | needs GPU or a long CPU run |

## 500-Example Baseline Results

These are real results from this repo.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Ex/sec |
|---|---:|---:|---:|---:|---:|
| Lead-1 | 0.2377 | 0.0785 | 0.1734 | 0.0554 | 9881.2667 |
| Lead-2 | 0.2981 | 0.1126 | 0.2058 | 0.1096 | 10618.0783 |
| Lead-3 | 0.3014 | 0.1187 | 0.2070 | 0.1612 | 10151.2331 |

## 50-Example DistilBART Review Run

This run is real too, but it is small.

| Model | Examples | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency sec/ex |
|---|---:|---:|---:|---:|---:|---:|
| DistilBART CNN | 50 | 0.3468 | 0.1547 | 0.2583 | 0.1219 | 14.3736 |

At this speed, 500 DistilBART examples would take about 2 hours on this CPU.
That estimate does not include BART-large-CNN or PEGASUS.

## What A Stronger Run Should Include

For a better resume claim, run at least 500 examples.

Models:

- Lead-1
- Lead-2
- Lead-3
- DistilBART-CNN
- BART-large-CNN
- PEGASUS CNN/DailyMail, if the machine can handle it

Metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum
- compression ratio
- latency per example
- examples per second
- total time

## Command For A GPU Machine

Use this on a machine with CUDA:

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

For 1,000 examples, change `--sample-size 500` to `--sample-size 1000`.

## Manual Review

The review sheet is already here:

```text
outputs/error_analysis/distilbart_50_manual_review_template.csv
```

Mark these columns with `yes` when needed:

- missing key fact
- hallucination
- entity error
- repetition
- over-compression

After filling it, run the summary script.

Don't claim an error rate until the sheet is filled.

## Content Discovery

The repo already creates tags and nearby items from summaries:

```text
outputs/content_discovery/distilbart_50/
```

That is a useful start.

It is not proof that search got better. To prove that, I need topic labels,
genre labels, editorial sections, or search queries.

## Future Bullets

These are future bullets, not current claims:

- Built a CNN/DailyMail benchmark with Lead baselines, DistilBART, and
  BART-large-CNN over 500 examples.
- Reached ROUGE-1 0.423, ROUGE-2 0.201, and ROUGE-L 0.311 with BART-large-CNN.
- Finished a 50-example manual review with error flags under 8 percent.

Use them only if the repo later has the exact files to support them.
