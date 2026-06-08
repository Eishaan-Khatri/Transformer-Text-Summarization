# Picking CV Bullets For This Project

This file helps choose CV points that are true and still worth reading.

The easy mistake is to make the project sound bigger than it is. I don't want
that. A recruiter or interviewer should be able to open the repo and find the
same proof.

## The Honest Story

The story is not:

> I built a company-ready summarizer.

The better story is:

> I rebuilt a summarization project properly. It has baselines, Transformer
> inference, metrics, charts, review files, and a small Transformer from
> scratch.

That is enough. It shows real work without pretending the project is something
else.

## What The CV Should Show

The CV should make three things clear:

1. I can run an NLP experiment from start to finish.
2. I understand Transformer parts, not just the Hugging Face API.
3. I save numbers and outputs instead of making loose claims.

## Proof In The Repo

| Proof | File |
|---|---|
| Small CNN/DailyMail benchmark | `outputs/tables/benchmark_summary.csv` |
| 500-example Lead baseline run | `outputs/tables/baseline_500_summary.csv` |
| 50-example DistilBART review run | `outputs/metrics/distilbart_50_review/` |
| Charts | `outputs/figures/` |
| Manual review sheet | `outputs/error_analysis/distilbart_50_manual_review_template.csv` |
| Summary tags and neighbors | `outputs/content_discovery/distilbart_50/` |
| NumPy Transformer | `src/scratch_transformer.py` |
| Tests | `tests/` |

## Numbers I Can Defend

Small 24-example run:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression |
|---|---:|---:|---:|---:|
| Lead-1 | 0.2242 | 0.0787 | 0.1636 | 0.0558 |
| Lead-2 | 0.2860 | 0.1156 | 0.2110 | 0.1169 |
| Lead-3 | 0.3038 | 0.1212 | 0.2105 | 0.1727 |
| DistilBART CNN | 0.3470 | 0.1399 | 0.2442 | 0.1313 |

500-example baseline run:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression |
|---|---:|---:|---:|---:|
| Lead-1 | 0.2377 | 0.0785 | 0.1734 | 0.0554 |
| Lead-2 | 0.2981 | 0.1126 | 0.2058 | 0.1096 |
| Lead-3 | 0.3014 | 0.1187 | 0.2070 | 0.1612 |

50-example DistilBART review run:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency sec/ex |
|---|---:|---:|---:|---:|---:|
| DistilBART CNN | 0.3468 | 0.1547 | 0.2583 | 0.1219 | 14.3736 |

The DistilBART number is useful, but it is still small. I would call it a
checked CPU run, not a final benchmark.

## Best 3 CV Bullets Right Now

Use these if this project gets three bullets:

- Built a CNN/DailyMail summarization benchmark with Hugging Face
  Transformers, Lead-1/2/3 baselines, ROUGE metrics, compression tracking, and
  generated charts.
- Measured DistilBART on a checked 50-example CNN/DailyMail CPU run, reaching
  ROUGE-1 0.3468, ROUGE-2 0.1547, ROUGE-L 0.2583, and 12.2 percent average
  compression.
- Wrote a compact NumPy encoder-decoder Transformer covering positional
  encoding, multi-head attention, causal masking, cross-attention, layer
  normalization, and output logits.

## Shorter Version

Use this if the CV has less space:

- Built a CNN/DailyMail summarization benchmark with Lead baselines, ROUGE
  metrics, compression tracking, and charts.
- Measured DistilBART on a checked 50-example CPU run with ROUGE-1 0.3468,
  ROUGE-2 0.1547, ROUGE-L 0.2583, and 12.2 percent compression.
- Wrote a small NumPy Transformer covering attention, masking,
  encoder-decoder flow, and output logits.

## One-Line Project Summary

Transformer text summarization benchmark with Hugging Face inference, Lead
baselines, ROUGE evaluation, generated charts, review files, and a NumPy
Transformer from scratch.

## What I Should Not Put On The CV

Don't write:

- "Company-ready summarization system"
- "Best summarizer in the field"
- "Improved search relevance"
- "Deployed content recommendation pipeline"
- "Trained Transformer from scratch on CNN/DailyMail"
- "Full CNN/DailyMail benchmark"
- "Batch size 32" unless a real run proves it
- "BART-large-CNN achieved ROUGE-1 0.423" unless that run exists in
  `outputs/metrics/`

## What Would Make The CV Point Stronger

The next upgrade is clear:

1. Run 500 or 1,000 examples on GPU.
2. Add BART-large-CNN.
3. Add PEGASUS if the runtime is manageable.
4. Fill the 50-row manual review sheet.
5. Add labelled retrieval or a query set before saying summaries help search.

Until then, this project is a good support project. It should not be the main
headline of the resume.
