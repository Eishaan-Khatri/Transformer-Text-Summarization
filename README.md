# Text Summarization with Transformers

Rebuilt NLP project for transformer-based summarization, content representation,
batch inference, and ROUGE/compression evaluation.

> Status: rebuilt public version. The original repository was empty, so this
> project is reconstructed as a stronger and more defensible version. Do not
> claim old results unless they are reproduced by the full benchmark.

## What This Project Shows

- Pretrained transformer summarization benchmark design for CNN/DailyMail.
- Lead baseline for comparison.
- ROUGE-1, ROUGE-2, ROUGE-L, and compression-ratio utilities.
- Batched inference script for Hugging Face encoder-decoder models.
- A small NumPy encoder-decoder Transformer built from scratch to demonstrate
  positional encoding, masking, multi-head attention, cross-attention, and
  decoder logits.
- Content-platform mapping for compact metadata, browsing cards, and cold-start
  item understanding.

## Current Local Verification

The current machine does not have PyTorch, Transformers, Datasets, Evaluate, or
ROUGE installed. The checked-in lightweight layer is dependency-light and has
been designed so the project still has runnable evidence before heavy model
downloads.

Run local tests:

```powershell
python -m unittest discover -s tests
```

Run the lightweight demo:

```powershell
python -m src.run_lightweight_demo
```

The demo writes:

- `outputs/metrics/lightweight_demo_metrics.json`
- `outputs/metrics/lightweight_demo_metrics.csv`
- `outputs/examples/sample_summaries.jsonl`
- `reports/final_report.md`

These outputs prove the metric, baseline, batching, and scratch Transformer
components run. They are not the CNN/DailyMail benchmark.

## Full CNN/DailyMail Benchmark

Install the full dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run a small benchmark:

```powershell
python -m src.hf_benchmark --model facebook/bart-large-cnn --split test --sample-size 128 --batch-size 8
```

Run a stronger throughput comparison after confirming hardware capacity:

```powershell
python -m src.hf_benchmark --model facebook/bart-large-cnn --split test --sample-size 1024 --batch-size 32
python -m src.hf_benchmark --model google/pegasus-cnn_dailymail --split test --sample-size 1024 --batch-size 16
```

Outputs are written under `outputs/metrics/full_benchmark/`.

## Repository Map

| Path | Purpose |
|---|---|
| `src/metrics.py` | ROUGE and compression-ratio utilities |
| `src/data.py` | JSONL loading, normalization, and batching |
| `src/lead_baseline.py` | Lead sentence baseline |
| `src/scratch_transformer.py` | NumPy mini encoder-decoder Transformer |
| `src/hf_benchmark.py` | Full Hugging Face CNN/DailyMail benchmark runner |
| `src/run_lightweight_demo.py` | Local demo that works without heavy ML dependencies |
| `tests/` | Unit tests for metrics, batching, and scratch Transformer behavior |
| `docs/` | Methodology, limitations, full benchmark runbook, and interview notes |
| `outputs/` | Generated demo and benchmark outputs |
| `reports/` | Generated final report |

## Safe Portfolio Claim After Full Benchmark

Use this only after running the full benchmark and replacing metrics with the
measured values:

> Built a reproducible CNN/DailyMail summarization benchmark using Hugging Face
> transformer models, batched inference, ROUGE/compression evaluation, and a
> scratch-built mini Transformer to demonstrate attention, masking, and
> encoder-decoder internals.

## Claims Not To Make

- Do not say this project is state of the art.
- Do not say the scratch Transformer competes with BART or PEGASUS.
- Do not claim search relevance improved unless a retrieval experiment is run.
- Do not reuse the old `ROUGE-L = 0.7` or `5 million words in under 20 seconds`
  wording without hard evidence.

