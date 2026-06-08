# Interview Notes

Short version:

> I built a summarization benchmark on CNN/DailyMail and compared Lead baselines
> with a DistilBART summarizer. I measured ROUGE, compression ratio, and CPU
> throughput. I also wrote a small Transformer from scratch in NumPy so I could
> explain attention, masking, and encoder-decoder flow.

## If Asked About Results

The checked run used 24 test examples on CPU.

DistilBART got:

- ROUGE-1: 0.3470
- ROUGE-2: 0.1399
- ROUGE-L: 0.2442
- compression ratio: 0.1313
- throughput: 0.0731 examples/sec

Lead-3 got:

- ROUGE-1: 0.3038
- ROUGE-2: 0.1212
- ROUGE-L: 0.2105

So DistilBART was better on ROUGE, but much slower on CPU.

## If Asked Why I Built A Transformer From Scratch

Because using pretrained models alone does not show the internals. The NumPy
version helped me understand positional encoding, attention scores, causal
masking, cross-attention, and decoder logits.

## What Not To Overclaim

- I did not train a competitive summarizer from scratch.
- I did not run the full CNN/DailyMail test split.
- I did not prove factual consistency.
- I did not prove search or recommendation improvement.
