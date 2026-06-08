# Transformer From Scratch

The from-scratch component exists to prove architecture understanding, not to
claim benchmark superiority.

## Implemented Components

| Component | File | Why It Matters |
|---|---|---|
| Token-position representation | `src/scratch_transformer.py` | Shows how sequence order enters the model |
| Scaled dot-product attention | `src/scratch_transformer.py` | Core attention operation |
| Causal mask | `src/scratch_transformer.py` | Prevents decoder positions from seeing future tokens |
| Multi-head attention | `src/scratch_transformer.py` | Lets attention use multiple representation subspaces |
| Encoder layer | `src/scratch_transformer.py` | Builds contextual source representations |
| Decoder layer | `src/scratch_transformer.py` | Combines prior target tokens with encoded source context |
| Vocabulary logits | `src/scratch_transformer.py` | Produces next-token scores |

## Why NumPy

NumPy keeps the implementation inspectable and runnable in the current local
environment. A PyTorch version would be better for training, but it would also
hide some of the mechanics behind framework modules.

## Honest Interview Explanation

If asked, describe it this way:

> I built a small encoder-decoder Transformer from scratch to understand and
> demonstrate the internal mechanics: positional encoding, multi-head attention,
> causal masking, cross-attention, and decoder logits. For actual summarization
> quality, I benchmark pretrained/fine-tuned models like BART and PEGASUS,
> because training a competitive summarizer from scratch is not realistic on a
> small student compute budget.

