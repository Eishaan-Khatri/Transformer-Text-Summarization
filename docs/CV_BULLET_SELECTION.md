# CV Bullet Selection: Transformer Text Summarization

This file is for choosing the best 3 CV points from the project without making
the project sound bigger than it is.

The strongest story is not:

> I built a company-ready summarizer.

The stronger and safer story is:

> I rebuilt a full summarization workflow: baselines, Transformer inference,
> metrics, charts, and a small Transformer from scratch so I could explain the
> internals.

## What The CV Needs To Show

A good CV entry should prove three things fast:

1. I can build an end-to-end NLP experiment, not just run one command.
2. I understand Transformer architecture beyond the Hugging Face API.
3. I can measure results honestly and show them with tables/charts.

## Evidence Available

| Evidence | Where it is shown | How strong it is |
|---|---|---|
| CNN/DailyMail summarization benchmark | `src/hf_benchmark.py`, `outputs/tables/benchmark_summary.csv` | Strong |
| Lead-1, Lead-2, Lead-3 baselines | `src/build_results.py`, output JSON/CSV files | Strong |
| DistilBART inference with Hugging Face Transformers | `src/hf_benchmark.py`, `outputs/metrics/full_benchmark/` | Strong |
| ROUGE, compression ratio, elapsed time, throughput | `src/metrics.py`, `outputs/tables/benchmark_summary.csv` | Strong |
| Charts for ROUGE, compression, CPU throughput | `outputs/figures/` | Strong |
| 500-example Lead baseline run | `outputs/tables/baseline_500_summary.csv` | Strong |
| 50-example DistilBART review run | `outputs/metrics/distilbart_50_review/` | Strong for review, not final benchmark |
| Manual error review sheet | `outputs/error_analysis/distilbart_50_manual_review_template.csv` | Created, not filled yet |
| Content-discovery tags/neighbors | `outputs/content_discovery/distilbart_50/` | Useful bridge, not search proof |
| NumPy Transformer from scratch | `src/scratch_transformer.py` | Strong |
| Unit tests | `tests/` | Strong |
| Search/retrieval improvement | Not tested yet | Weak, do not claim |
| Large full-test benchmark | Not run yet | Weak, do not claim |
| Production deployment | Not built | Do not claim |

## Checked Numbers

The main README run used 24 CNN/DailyMail test examples on CPU.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Ex/sec |
|---|---:|---:|---:|---:|---:|
| Lead-1 | 0.2242 | 0.0787 | 0.1636 | 0.0558 | 5957.7003 |
| Lead-2 | 0.2860 | 0.1156 | 0.2110 | 0.1169 | 6679.8408 |
| Lead-3 | 0.3038 | 0.1212 | 0.2105 | 0.1727 | 7350.4640 |
| DistilBART CNN | 0.3470 | 0.1399 | 0.2442 | 0.1313 | 0.0731 |

Important warning: these are checked numbers from a small CPU run. They are
good enough to prove the pipeline works. They are not enough to claim a final
benchmark result on the full CNN/DailyMail test set.

The larger baseline run used 500 examples:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression |
|---|---:|---:|---:|---:|
| Lead-1 | 0.2377 | 0.0785 | 0.1734 | 0.0554 |
| Lead-2 | 0.2981 | 0.1126 | 0.2058 | 0.1096 |
| Lead-3 | 0.3014 | 0.1187 | 0.2070 | 0.1612 |

The 50-example DistilBART review run got:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency sec/ex |
|---|---:|---:|---:|---:|---:|
| DistilBART CNN | 0.3468 | 0.1547 | 0.2583 | 0.1219 | 14.3736 |

## Best 3 CV Points

Use these if the CV has space for 3 bullets under this project.

### Point 1: End-To-End NLP Evaluation

**CV bullet:**

- Built an end-to-end CNN/DailyMail summarization benchmark with Hugging Face
  Transformers, Lead-1/2/3 baselines, ROUGE evaluation, compression tracking,
  and generated result charts.

**Why this is strong:**

This shows the project is more than a model call. It includes baselines,
metrics, outputs, and reporting.

**Interview backup:**

I compared DistilBART with Lead baselines because news articles often put the
main facts near the start. Without those baselines, a Transformer result can
look better than it really is.

### Point 2: Measured Transformer Inference

**CV bullet:**

- Evaluated DistilBART summarization on a checked 50-example CNN/DailyMail CPU
  run, reaching ROUGE-1 0.3468, ROUGE-2 0.1547, ROUGE-L 0.2583, and 12.2
  percent average compression.

**Why this is strong:**

This gives numbers. It shows the model was run, measured, and compared instead
of only described.

**Risk:**

Do not remove the word "checked" if asked about the result. The current neural
model run is still small. If someone asks, say the repo now has 500-example
baselines and a 50-example DistilBART CPU run, and that the next step is a
500-example neural run on GPU.

**Interview backup:**

DistilBART had the best ROUGE score in this run, but CPU throughput was slow.
Lead-3 was weaker on ROUGE but much faster because it only slices sentences.

### Point 3: Transformer From Scratch

**CV bullet:**

- Implemented a compact encoder-decoder Transformer in NumPy with positional
  encoding, scaled dot-product attention, multi-head attention, causal masking,
  cross-attention, feed-forward layers, layer normalization, and output logits.

**Why this is strong:**

This is the deepest technical point. It tells a reviewer that I can explain how
the model works inside, not only how to call a pretrained summarization model.

**Interview backup:**

The scratch model is not trained to beat DistilBART. It is there to show the
architecture: how attention scores are made, why causal masking is needed in
the decoder, and how cross-attention connects the decoder to the source article.

## Final CV Version

Use this if the project gets 3 bullets:

- Built an end-to-end CNN/DailyMail summarization benchmark with Hugging Face
  Transformers, Lead-1/2/3 baselines, ROUGE evaluation, compression tracking,
  and generated result charts.
- Evaluated DistilBART summarization on a checked 50-example CNN/DailyMail CPU
  run, reaching ROUGE-1 0.3468, ROUGE-2 0.1547, ROUGE-L 0.2583, and 12.2
  percent average compression.
- Implemented a compact encoder-decoder Transformer in NumPy with positional
  encoding, scaled dot-product attention, multi-head attention, causal masking,
  cross-attention, feed-forward layers, layer normalization, and output logits.

## Shorter CV Version

Use this if the CV is tight and each bullet must be shorter:

- Built a CNN/DailyMail summarization benchmark with Hugging Face Transformers,
  Lead baselines, ROUGE metrics, compression tracking, and result charts.
- Measured DistilBART on a checked 50-example CPU run, reaching ROUGE-1 0.3468,
  ROUGE-2 0.1547, ROUGE-L 0.2583, and 12.2 percent average compression.
- Wrote a compact NumPy Transformer covering attention, masking,
  encoder-decoder flow, layer normalization, and output logits.

## One-Line Project Summary

Use this for a portfolio card or resume project title line:

Transformer text summarization benchmark with Hugging Face inference, ROUGE
evaluation, baseline comparison, charts, and a NumPy Transformer implementation
from scratch.

## What Not To Put On The CV

Do not write:

- "Company-ready summarization system"
- "Best summarizer in the field"
- "Improved search relevance"
- "Deployed content recommendation pipeline"
- "Trained Transformer from scratch on CNN/DailyMail"
- "Full CNN/DailyMail benchmark"
- "Batch size 32" unless a real run proves it
- "ROUGE-1 0.40 / ROUGE-2 0.17 / ROUGE-L 0.34" unless the repo contains a real
  run with those exact numbers
- "BART-large-CNN achieved ROUGE-1 0.423" unless the 500-example BART run is
  actually saved in `outputs/metrics/`

## If I Want A Stronger CV Claim Later

To upgrade the CV point, I should run:

1. Larger sample size, at least a few hundred examples.
2. GPU inference with a stronger model such as BART-large-CNN or PEGASUS.
3. The same baselines on the same rows.
4. A saved metrics table with timestamp, model, dataset split, and hardware.
5. A factuality check or manual error review.
6. A small retrieval test if I want to claim search/discovery relevance.

Only after that should the CV claim move from "checked CPU run" to a stronger
benchmark claim.
