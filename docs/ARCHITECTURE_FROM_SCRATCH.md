# Transformer From Scratch

I added this because I didn't want the project to be only a Hugging Face model
call.

The file is:

```text
src/scratch_transformer.py
```

It is a small encoder-decoder Transformer written in NumPy.

Think of the model like a student reading an article and then writing a short
answer. The encoder reads the article. The decoder writes the answer one token
at a time. Attention is the part that helps the model decide which words matter
right now.

## What Is Inside

| Part | Simple meaning |
|---|---|
| Positional encoding | Tells the model word order |
| Scaled dot-product attention | Scores which tokens should matter more |
| Causal mask | Stops the decoder from peeking at future words |
| Multi-head attention | Lets the model look at text in several ways |
| Encoder layer | Builds article features |
| Decoder layer | Uses old output tokens plus article context |
| Cross-attention | Lets the decoder look back at the article |
| Output projection | Produces scores for the next token |

## What It Is For

This scratch model is for understanding.

It is not trained to beat DistilBART. Training a serious summarizer from
scratch would need far more data, time, and compute.

The useful part is that I can point to the code and explain where masking
happens, why attention is scaled, and how the decoder uses the article while
writing the summary.
