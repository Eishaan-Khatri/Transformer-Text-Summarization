# Methodology

## Project Goal

The project is designed to show both practical summarization evaluation and
Transformer architecture understanding.

The practical layer benchmarks pretrained/fine-tuned summarizers on
CNN/DailyMail. The architecture layer builds a small encoder-decoder Transformer
from scratch in NumPy so attention, masks, positions, and logits are inspectable.

## Benchmark Layer

Recommended benchmark components:

1. Lead-3 or Lead-2 baseline.
2. `facebook/bart-large-cnn`.
3. `google/pegasus-cnn_dailymail`.
4. Optional smaller model for speed comparison.

Measured outputs:

- ROUGE-1.
- ROUGE-2.
- ROUGE-L or ROUGE-LSUM.
- Compression ratio: generated summary tokens divided by input article tokens.
- Batch size.
- Runtime.
- Examples per second.
- Device: CPU or CUDA.

## Scratch Transformer Layer

The scratch model is intentionally small. It demonstrates:

- sinusoidal positional encoding,
- causal decoder mask,
- scaled dot-product attention,
- multi-head attention,
- encoder self-attention,
- decoder self-attention,
- encoder-decoder cross-attention,
- feed-forward blocks,
- layer normalization,
- vocabulary-sized decoder logits.

It is not presented as a competitive summarization model. It is an architecture
inspection component.

## Why This Is Stronger Than A Notebook

A notebook-only project often proves that a tutorial was followed. This rebuild
separates reusable code, tests, benchmark scripts, generated outputs, and
claim-control documentation. That makes the project easier to defend in an
interview.

