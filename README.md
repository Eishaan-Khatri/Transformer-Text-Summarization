# Transformer Text Summarization

Long articles take time to read. A summarizer tries to give the short version
without losing the main point. That sounds simple, but it is easy to make a bad
summary: it can skip the important part, repeat too much, or sound right while
missing context.

I built this project to check that problem in a practical way. I used news
articles from CNN/DailyMail, compared a Transformer summarizer with simple
baselines, measured the results, and plotted the scores.

I also built a small Transformer in NumPy. That part is there so I can explain
what happens inside the model: attention, masking, encoder-decoder flow, and
the final token scores.

Think of it like this: if a friend asks, "What was that long article about?",
the model should give a short answer that still keeps the important facts. This
project checks how close we get to that.

## Tiny Example

This is a made-up example, just to explain the idea.

If a long article says a city opened a bus-delay dashboard, found two routes
causing most evening delays, and plans to add more buses there, a weak summary
might only say:

> The city opened a bus dashboard.

That is short, but it misses the useful part. A better summary would keep the
main point:

> The city is tracking bus delays and found two routes causing most evening
> problems, so it plans to add buses there.

That is what this project is really testing: not just "can it be short?", but
"does it keep the part people came for?"

## How I Checked It

Machine/run details:

- Dataset: `abisee/cnn_dailymail`
- Config: `1.0.0`
- Split: `test`
- Sample: first 24 test examples
- Device: CPU
- Transformer model: `sshleifer/distilbart-cnn-6-6`
- Batch size for DistilBART: 2

## Measured Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Ex/sec |
|---|---:|---:|---:|---:|---:|
| Lead-1 baseline | 0.2242 | 0.0787 | 0.1636 | 0.0558 | 5957.7003 |
| Lead-2 baseline | 0.2860 | 0.1156 | 0.2110 | 0.1169 | 6679.8408 |
| Lead-3 baseline | 0.3038 | 0.1212 | 0.2105 | 0.1727 | 7350.4640 |
| DistilBART CNN | 0.3470 | 0.1399 | 0.2442 | 0.1313 | 0.0731 |

DistilBART got the best ROUGE score in this run. It was also much slower on
CPU. Lead-3 is still worth checking because news articles often put the main
facts near the start.

The throughput chart uses a log scale. The lead baselines are just sentence
slicing, so their speed is not directly comparable to neural model inference.

## Charts

![ROUGE comparison](outputs/figures/rouge_comparison.png)

![Compression ratio](outputs/figures/compression_ratio.png)

![CPU throughput](outputs/figures/throughput.png)

## Project Structure

| Path | Purpose |
|---|---|
| `src/hf_benchmark.py` | Runs the DistilBART benchmark on CNN/DailyMail rows |
| `src/build_results.py` | Builds baseline results, summary tables, charts, and report |
| `src/scratch_transformer.py` | Mini Transformer from scratch in NumPy |
| `src/metrics.py` | ROUGE-style and compression utilities |
| `tests/` | Unit tests for metrics, data loading, and Transformer internals |
| `outputs/tables/benchmark_summary.csv` | Main result table |
| `outputs/figures/` | Generated charts |
| `reports/final_report.md` | Short written result summary |

## Run It

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run tests:

```powershell
python -m unittest discover -s tests
```

Run the model benchmark:

```powershell
python -m src.hf_benchmark --model sshleifer/distilbart-cnn-6-6 --dataset abisee/cnn_dailymail --config 1.0.0 --split test --sample-size 24 --batch-size 2 --max-new-tokens 96
```

Build tables and charts:

```powershell
python -m src.build_results --run-baselines --sample-size 24
```

## What I Would Improve Next

- Run the same comparison on a larger test sample.
- Try BART-large-CNN or PEGASUS on a GPU.
- Add factuality checks, because ROUGE mostly checks word overlap.
- Add a small search/retrieval experiment to test whether summaries help item
  discovery.

## Notes

Compression ratio here means:

```text
summary tokens / article tokens
```

A lower compression ratio is not automatically better. A very short summary can
miss important facts.
