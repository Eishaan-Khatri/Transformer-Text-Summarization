# Summarization Results

I ran this on `abisee/cnn_dailymail` with `24` test examples.

This is a small CPU run. It checks that the code, metrics, baselines, and charts work together.

It is not a final benchmark.

## Metrics

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency sec/ex | Ex/sec |
|---|---:|---:|---:|---:|---:|---:|
| Lead-1 | 0.2242 | 0.0787 | 0.1636 | 0.0558 | 0.0002 | 5957.7003 |
| Lead-2 | 0.2860 | 0.1156 | 0.2110 | 0.1169 | 0.0001 | 6679.8408 |
| Lead-3 | 0.3038 | 0.1212 | 0.2105 | 0.1727 | 0.0001 | 7350.4640 |
| DistilBART CNN | 0.3470 | 0.1399 | 0.2442 | 0.1313 | 13.6741 | 0.0731 |

## Charts

![ROUGE comparison](../outputs/figures/rouge_comparison.png)

![Compression ratio](../outputs/figures/compression_ratio.png)

![CPU throughput](../outputs/figures/throughput.png)

![Latency per example](../outputs/figures/latency_per_example.png)

## What I Take From This Run

- DistilBART scored best on ROUGE in this small run.
- It was also slow on CPU.
- Lead-3 is still useful because news articles often start with the main facts.
- Compression and ROUGE should be read together. A shorter summary is not automatically better.
- The next serious step is a 500-example neural run on GPU.
