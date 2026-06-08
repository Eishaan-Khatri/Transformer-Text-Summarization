# Interview Notes

## Short Answer

> I rebuilt a text summarization project around CNN/DailyMail. I compared Lead
> baselines with DistilBART, measured ROUGE and compression, saved charts, and
> wrote a small Transformer in NumPy so I could explain the model internals.

## If They Ask About Results

The main README run is small: 24 examples on CPU.

DistilBART got:

- ROUGE-1: 0.3470
- ROUGE-2: 0.1399
- ROUGE-L: 0.2442
- compression ratio: 0.1313
- speed: 0.0731 examples/sec

Lead-3 got:

- ROUGE-1: 0.3038
- ROUGE-2: 0.1212
- ROUGE-L: 0.2105

So DistilBART scored better, but it was much slower on CPU.

The repo also has 500-example Lead baseline outputs and a 50-example DistilBART
review run. I would still want a GPU run before calling it a serious benchmark.

## If They Ask Why Lead Baselines Matter

News articles often put the main facts at the start.

So a simple baseline like "take the first three sentences" can do surprisingly
well. If a Transformer can't beat that, the model result is not very impressive.

## If They Ask Why I Built A Transformer From Scratch

Using a pretrained model shows that I can run the tool.

Writing a small Transformer shows that I understand the moving parts: attention
scores, masks, encoder-decoder flow, cross-attention, and output logits.

The scratch model is for learning and explanation. It is not a large trained
summarizer.

## What I Should Not Say

- I didn't train a serious summarizer from scratch.
- I didn't run BART-large-CNN or PEGASUS yet.
- I didn't prove summaries improve search.
- I didn't finish the manual hallucination/error review yet.
