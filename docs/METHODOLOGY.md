# Methodology

I kept the project simple on purpose. The goal was not to train a huge model
from scratch. The goal was to build a complete summarization experiment that I
could explain line by line.

## Data

I used the CNN/DailyMail test split through the accessible Hugging Face mirror:

```text
abisee/cnn_dailymail, config 1.0.0
```

The current checked run uses 24 test examples. That is small, but it is enough
to check whether the pipeline works and to compare baselines with a transformer
model on the same rows.

## Models

I compared:

- Lead-1 baseline
- Lead-2 baseline
- Lead-3 baseline
- `sshleifer/distilbart-cnn-6-6`

The lead baselines are not filler. News articles often put key facts in the
first few sentences, so a serious summarization experiment should check them.

## Metrics

I measured:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum
- compression ratio
- examples per second
- elapsed time

Compression ratio is:

```text
summary tokens / article tokens
```

## Transformer From Scratch

The NumPy Transformer is separate from the benchmark model. It includes:

- sinusoidal positional encoding,
- scaled dot-product attention,
- multi-head attention,
- causal decoder masking,
- encoder self-attention,
- decoder self-attention,
- encoder-decoder cross-attention,
- feed-forward layers,
- layer normalization.

I use it to show that I understand the architecture. I do not use it as the
main summarization model.
