# Methodology

I kept the project practical.

The goal was not to train a huge model from scratch. The goal was to rebuild the
summarization workflow in a way I can explain: data, baselines, model inference,
metrics, charts, and limits.

## Data

I used CNN/DailyMail test rows from:

```text
abisee/cnn_dailymail, config 1.0.0
```

Each row has a news article and a reference summary.

## Models

I compared:

- Lead-1
- Lead-2
- Lead-3
- `sshleifer/distilbart-cnn-6-6`

Lead baselines are simple: take the first 1, 2, or 3 sentences.

They are still worth checking because news stories often put the main point
near the start.

## Metrics

I saved:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum
- compression ratio
- latency per example
- examples per second
- total time

Compression ratio means:

```text
summary tokens / article tokens
```

## Review And Discovery

The repo also has:

- a 50-row manual error review sheet,
- summary tags,
- similar-item candidates.

Those pieces help connect summarization to content discovery. They do not prove
search is better yet.

## Transformer From Scratch

The NumPy Transformer is separate from the benchmark model.

It includes positional encoding, attention, masking, cross-attention,
feed-forward layers, layer normalization, and output logits.

I use it to explain the architecture. I don't use it as the main summarizer.
