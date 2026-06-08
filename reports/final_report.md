# Summarization Results

I ran this on `abisee/cnn_dailymail` with `24` test examples. This is a small CPU run. I use it to check the pipeline and compare methods on the same examples, not to pretend it is a final benchmark.

## Metrics

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Ex/sec |
|---|---:|---:|---:|---:|---:|
| Lead-1 | 0.2242 | 0.0787 | 0.1636 | 0.0558 | 5957.7003 |
| Lead-2 | 0.2860 | 0.1156 | 0.2110 | 0.1169 | 6679.8408 |
| Lead-3 | 0.3038 | 0.1212 | 0.2105 | 0.1727 | 7350.4640 |
| DistilBART CNN | 0.3470 | 0.1399 | 0.2442 | 0.1313 | 0.0731 |

## Charts

![ROUGE comparison](../outputs/figures/rouge_comparison.png)

![Compression ratio](../outputs/figures/compression_ratio.png)

![CPU throughput](../outputs/figures/throughput.png)

## What I take from this run

- DistilBART got the best ROUGE numbers in this small run, but it was slow on CPU.
- Lead baselines are useful because CNN/DailyMail articles often put important facts near the start.
- Baseline throughput is simple sentence slicing, so it should not be read as model inference speed.
- Compression ratio matters separately from ROUGE. A shorter summary is not automatically better.
- I would run a larger sample before using these numbers as a main project claim.
