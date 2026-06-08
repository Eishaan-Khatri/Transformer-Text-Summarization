# Limitations

## Rebuild Status

The public GitHub repository was empty when checked. This version is a rebuilt
project, not recovered original source code.

## Dependency Status

The current local environment used for the first build did not contain PyTorch,
Transformers, Datasets, Evaluate, or ROUGE. The lightweight demo runs locally,
but full benchmark claims require installing those dependencies and running the
CNN/DailyMail benchmark.

## Metric Limits

ROUGE measures lexical overlap. It does not prove factual correctness, usefulness
for search, or reader satisfaction. Good summarization evaluation should also
inspect entity errors, hallucinations, repetition, and missing context.

## Compression Limits

Compression ratio must be defined clearly. This project uses:

```text
generated_summary_tokens / input_article_tokens
```

A lower number is not always better. A summary can be short but misleading.

## Content-Platform Limits

Compact summaries can help browsing cards and cold-start metadata, but actual
search or recommendation lift requires a retrieval/ranking experiment or online
product data. This project should not claim business impact without that test.

## Scratch Transformer Limits

The from-scratch Transformer is a small NumPy implementation for learning and
explanation. It is not a production model and should not be compared directly
against pretrained models trained on large corpora.

