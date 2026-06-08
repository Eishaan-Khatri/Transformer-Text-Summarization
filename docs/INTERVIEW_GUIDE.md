# Interview Guide

## Short Explanation

I rebuilt this as a summarization benchmark and architecture project. The main
pipeline evaluates pretrained transformer summarizers on CNN/DailyMail using
ROUGE, compression ratio, and batch-inference throughput. Separately, I built a
small Transformer from scratch in NumPy to show I understand attention, masking,
positional encoding, and encoder-decoder mechanics.

## Why Not Train A Big Model From Scratch?

Training a competitive CNN/DailyMail summarizer from scratch is compute-heavy
and not the honest point of this project. The honest split is:

- pretrained models for quality and realistic inference,
- scratch Transformer for architecture understanding.

## What I Would Improve Next

- Run the full CNN/DailyMail test benchmark on GPU.
- Add DistilBART or another smaller model for speed/quality tradeoff.
- Add hallucination and entity-error analysis.
- Add a retrieval experiment to test whether summaries improve search snippets.

## Unsafe Answers To Avoid

- "I built a state-of-the-art summarizer."
- "My scratch Transformer beat BART."
- "This improved search relevance."
- "ROUGE proves the summary is factually correct."

