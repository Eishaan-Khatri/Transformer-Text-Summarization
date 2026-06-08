# Text Summarization with Transformers - Local Demo Report

This report is generated from the dependency-light local demo, not the full CNN/DailyMail benchmark.

## Local Demo Metrics

- Records evaluated: 4
- Baseline: lead_2_baseline
- Mean ROUGE-1: 0.3847
- Mean ROUGE-2: 0.1209
- Mean ROUGE-L: 0.3362
- Mean compression ratio: 0.4695
- Scratch Transformer logits shape: [1, 4, 64]

## Interpretation

The checked-in run proves that the data, metric, baseline, and scratch Transformer components execute locally.
The public portfolio claim should use full benchmark outputs only after running `src.hf_benchmark` with the heavy ML dependencies installed.