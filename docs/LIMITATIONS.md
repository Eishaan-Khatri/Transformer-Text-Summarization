# Limitations

This project is useful, but it has limits. I would rather say them clearly than
make the project sound bigger than it is.

## Small Neural Run

The main DistilBART result in the README uses 24 CNN/DailyMail test examples.

That is enough to show the code runs and the metrics are wired correctly. It is
not enough for a final benchmark claim.

There is also a 50-example DistilBART run for review. That is better for manual
checking, but still small.

## CPU Only

This machine has no CUDA.

The 50-example DistilBART run took 718.95 seconds. That is about 14.37 seconds
per example.

So a 500-example neural run belongs on a GPU machine, not this CPU setup.

## ROUGE Misses Some Problems

ROUGE checks word overlap with the reference summary.

That helps, but it does not prove the summary is factually right. A summary can
share words with the reference and still miss the main fact or add a wrong
detail.

That is why the repo has a manual error review sheet.

## Compression Is Not Quality

Compression ratio means:

```text
summary tokens / article tokens
```

Shorter is not always better. A very short summary can lose the useful part.

## Scratch Transformer

The NumPy Transformer is for understanding the architecture.

It is not trained to beat DistilBART, BART, or PEGASUS.

## Search And Discovery

The repo creates summary tags and similar-item candidates.

That is a useful bridge to content discovery, but it does not prove search got
better. For that, I need labels, queries, or a real retrieval test.
