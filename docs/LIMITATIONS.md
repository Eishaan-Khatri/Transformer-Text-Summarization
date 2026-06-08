# Limitations

This project is useful, but it has limits.

## Small Sample

The checked result uses 24 CNN/DailyMail test examples. That is not enough for a
final benchmark claim. It is enough to show that the pipeline runs and that the
baselines/model are evaluated in the same way.

## CPU Run

The DistilBART run was CPU-only. It took 328.18 seconds for 24 examples, so the
throughput number is not a statement about what the model can do on GPU.

## ROUGE Is Limited

ROUGE measures overlap with the reference summary. It does not prove factual
correctness. A summary can get a decent ROUGE score and still miss a key fact or
introduce a wrong detail.

## Compression Is Not Quality

The DistilBART run had a compression ratio of 0.1313. That means the generated
summary was around 13 percent of the input length. This is useful, but a shorter
summary is not automatically better.

## Scratch Transformer

The scratch Transformer is a small NumPy implementation. It is for learning and
inspection, not for beating pretrained summarization models.

## Search/Discovery Claim

The project is relevant to content discovery because summaries can become item
metadata. I have not run a retrieval experiment yet, so I do not treat this as
proof that search gets better.
