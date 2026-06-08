# Transformer From Scratch

I added this part because I did not want the project to be only a Hugging Face
API call.

The file is:

```text
src/scratch_transformer.py
```

It implements a small encoder-decoder Transformer in NumPy.

## Parts Implemented

| Part | What it does |
|---|---|
| Positional encoding | Adds token order information |
| Scaled dot-product attention | Computes attention scores and weighted values |
| Causal mask | Stops the decoder from seeing future target tokens |
| Multi-head attention | Splits attention across multiple heads |
| Encoder layer | Builds source/article representations |
| Decoder layer | Uses previous target tokens and source context |
| Output projection | Produces vocabulary-sized logits |

## Why This Matters

Calling `pipeline("summarization")` is not enough to show understanding. This
implementation lets me explain where masking happens, why attention is scaled,
how cross-attention differs from self-attention, and what the decoder outputs.

It is not trained as a large summarization model. That would need much more
compute and data.
