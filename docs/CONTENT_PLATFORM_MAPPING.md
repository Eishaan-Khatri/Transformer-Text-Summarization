# Content Discovery Mapping

This project can connect to content discovery, but only in a careful way.

Imagine a news or story app with thousands of long items. A user won't open
everything. The app needs short signals: a title, a tag, a preview, or a small
summary that helps people decide what to read next.

That is where summarization can help.

But this repo does not prove search is better yet. It only builds the first
step: summaries, tags, and similar-item candidates.

## What Exists Now

Run this after a model has created summaries:

```powershell
python -m src.content_discovery `
  --examples outputs/metrics/distilbart_50_review/sshleifer__distilbart-cnn-6-6_examples.csv `
  --top-k 5 `
  --output-dir outputs/content_discovery/distilbart_50
```

It writes:

- `summary_tags.csv`
- `summary_neighbors.csv`
- `retrieval_summary.json`

## What It Means

The script uses generated summaries to make simple tags and find nearby items.

Fictional example:

If one summary is about "bus delays, city routes, evening traffic," a similar
item might be another transport story. That could help a content platform group
related items together.

## What Is Still Missing

To prove this helps discovery, I still need one of these:

- topic labels,
- genre labels,
- editorial sections,
- or a small query set.

Then I can check metrics like Recall@K or MRR.

Without labels or queries, this is a bridge, not proof.
