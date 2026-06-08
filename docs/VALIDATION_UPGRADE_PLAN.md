# Validation Upgrade Plan

This project should support a content-discovery story, but it should not
dominate the resume yet.

The repo is stronger now than it was, but the main weakness is still clear: the
Transformer model runs are small because this machine is CPU-only.

## What Is Proven Now

| Check | Status | Evidence |
|---|---|---|
| 500-example Lead baseline benchmark | Done | `outputs/tables/baseline_500_summary.csv` |
| 50-example DistilBART run | Done | `outputs/metrics/distilbart_50_review/sshleifer__distilbart-cnn-6-6_summary.json` |
| Latency per example | Done | summary JSON files and benchmark CSVs |
| 50-row manual review sheet | Created | `outputs/error_analysis/distilbart_50_manual_review_template.csv` |
| Summary tags and similar-item candidates | Created | `outputs/content_discovery/distilbart_50/` |
| Labelled retrieval metric | Not done | CNN/DailyMail rows used here do not expose category/genre labels |
| BART-large-CNN 500-example run | Not done | needs GPU or long CPU run |
| PEGASUS 500-example run | Not done | needs GPU or long CPU run |

## Actual Larger Baseline Results

These are real 500-example CNN/DailyMail baseline results from this repo.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency sec/ex | Ex/sec |
|---|---:|---:|---:|---:|---:|---:|
| Lead-1 | 0.2377 | 0.0785 | 0.1734 | 0.0554 | 0.0001 | 9881.2667 |
| Lead-2 | 0.2981 | 0.1126 | 0.2058 | 0.1096 | 0.0001 | 10618.0783 |
| Lead-3 | 0.3014 | 0.1187 | 0.2070 | 0.1612 | 0.0001 | 10151.2331 |

## Actual DistilBART Review Run

This is a real 50-example CPU run. It is useful for error review, but it is not
a final benchmark.

| Model | Examples | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency sec/ex | Ex/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| DistilBART CNN | 50 | 0.3468 | 0.1547 | 0.2583 | 0.1219 | 14.3736 | 0.0695 |

At this speed, 500 DistilBART examples would take about 2 hours on this CPU
before adding BART-large-CNN or PEGASUS.

## Target Benchmark

For a serious resume claim, run at least 500 examples. Better: 1,000 examples.

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
- elapsed time

## Commands For A 500-Example GPU Run

Use these commands on a machine with CUDA. Running this full suite on CPU will
be slow.

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

## Manual Error Review

The 50-row review sheet already exists:

```text
outputs/error_analysis/distilbart_50_manual_review_template.csv
```

Review categories:

- missing key fact
- hallucination
- entity error
- repetition
- over-compression

After filling the sheet, summarize it with:

```powershell
python -m src.summarize_error_review `
  --input outputs/error_analysis/distilbart_50_manual_review_template.csv `
  --output-json outputs/error_analysis/distilbart_50_manual_review_summary.json `
  --output-csv outputs/error_analysis/distilbart_50_manual_review_summary.csv
```

Do not claim "hallucination/error rate under 8%" until that filled review sheet
actually supports it.

## Content-Discovery Bridge

The bridge output already exists for the 50 DistilBART summaries:

```text
outputs/content_discovery/distilbart_50/summary_tags.csv
outputs/content_discovery/distilbart_50/summary_neighbors.csv
outputs/content_discovery/distilbart_50/retrieval_summary.json
```

What it does:

- creates tags from generated summaries,
- builds simple TF-IDF summary embeddings,
- retrieves similar items by summary similarity,
- saves top-k neighbors.

What it does not prove yet:

- search relevance improved,
- recommendations improved,
- category-level retrieval works.

To prove retrieval properly, the dataset needs reliable labels such as topic,
category, genre, or editorial section. The CNN/DailyMail rows used here do not
provide those labels through the current data source.

## Future CV Bullets Only If True

These bullets are good only after the matching evidence exists in the repo:

- Built a CNN/DailyMail summarization benchmark with Lead baselines,
  DistilBART, and BART-large-CNN over a 500-example evaluation run.
- Achieved ROUGE-1 0.423, ROUGE-2 0.201, and ROUGE-L 0.311 with
  BART-large-CNN at 21.6 percent average compression.
- Completed a 50-example manual error review and kept hallucination/error flags
  under 8 percent.

Right now, those bullets are targets, not claims.
