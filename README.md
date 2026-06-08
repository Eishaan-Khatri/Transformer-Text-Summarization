# Transformer Text Summarization

I built this project to test text summarization models on news articles and to
understand the Transformer architecture beyond just calling a pipeline.

The project has two parts:

1. a small CNN/DailyMail benchmark using ROUGE, compression ratio, and CPU
   throughput;
2. a mini encoder-decoder Transformer written in NumPy, so the attention and
   masking steps are visible.

I am not claiming this is a production summarizer or a state-of-the-art model.
The useful part is the full workflow: baselines, model inference, metrics,
charts, and implementation of the core Transformer blocks.

## What I Ran

Machine/run details:

- Dataset: `abisee/cnn_dailymail`
- Config: `1.0.0`
- Split: `test`
- Sample: first 24 test examples
- Device: CPU
- Transformer model: `sshleifer/distilbart-cnn-6-6`
- Batch size for DistilBART: 2

## Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Ex/sec |
|---|---:|---:|---:|---:|---:|
| Lead-1 baseline | 0.2242 | 0.0787 | 0.1636 | 0.0558 | 5957.7003 |
| Lead-2 baseline | 0.2860 | 0.1156 | 0.2110 | 0.1169 | 6679.8408 |
| Lead-3 baseline | 0.3038 | 0.1212 | 0.2105 | 0.1727 | 7350.4640 |
| DistilBART CNN | 0.3470 | 0.1399 | 0.2442 | 0.1313 | 0.0731 |

DistilBART gives the best ROUGE score in this run, but it is much slower on CPU.
The Lead-3 baseline is still a useful comparison because CNN/DailyMail articles
often put important facts early.

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
- Add factuality checks, because ROUGE only measures word overlap.
- Add a small search/retrieval experiment to test whether summaries help item
  discovery.

## Notes

Compression ratio here means:

```text
summary tokens / article tokens
```

A lower compression ratio is not automatically better. A very short summary can
miss important facts.
