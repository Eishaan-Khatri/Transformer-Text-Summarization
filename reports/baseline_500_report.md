# Summarization Results

I ran this on `abisee/cnn_dailymail` with `500` test examples. This is a small CPU run. I use it to check the pipeline and compare methods on the same examples, not to pretend it is a final benchmark.

## Metrics

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency sec/ex | Ex/sec |
|---|---:|---:|---:|---:|---:|---:|
| Lead-1 | 0.2377 | 0.0785 | 0.1734 | 0.0554 | 0.0001 | 9881.2667 |
| Lead-2 | 0.2981 | 0.1126 | 0.2058 | 0.1096 | 0.0001 | 10618.0783 |
| Lead-3 | 0.3014 | 0.1187 | 0.2070 | 0.1612 | 0.0001 | 10151.2331 |

## Charts

![ROUGE comparison](../outputs/figures/rouge_comparison.png)

![Compression ratio](../outputs/figures/compression_ratio.png)

![CPU throughput](../outputs/figures/throughput.png)

![Latency per example](../outputs/figures/latency_per_example.png)

## What I take from this run

- DistilBART got the best ROUGE numbers in this small run, but it was slow on CPU.
- Lead baselines are useful because CNN/DailyMail articles often put important facts near the start.
- Baseline throughput is simple sentence slicing, so it should not be read as model inference speed.
- Compression ratio matters separately from ROUGE. A shorter summary is not automatically better.
- I would run a larger sample before using these numbers as a main project claim.
