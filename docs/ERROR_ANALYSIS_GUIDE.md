# Error Analysis Guide

ROUGE is useful, but it is not enough.

A summary can share many words with the reference and still be wrong. It can
miss the main fact. It can mix up a name. It can add a detail that was never in
the article.

That is why this repo has a manual review sheet.

## Review Sheet

```text
outputs/error_analysis/distilbart_50_manual_review_template.csv
```

Each row shows:

- a short article excerpt,
- the reference summary,
- the model summary,
- error columns,
- notes.

## Error Types

Use `yes` when you see the error. Leave it blank when you don't.

| Error type | What to look for |
|---|---|
| `missing_key_fact` | The summary skipped the main thing |
| `hallucination` | The summary added something not in the article |
| `entity_error` | A name, place, date, number, or group is wrong |
| `repetition` | The summary repeats itself |
| `over_compression` | The summary is too short to be useful |

## How I Would Review A Row

Start with the reference summary. Then read the model summary.

If something feels missing or suspicious, check the article excerpt.

Mark the error columns only when you can defend the mark. If the issue is
unclear, leave a short note.

## After The Sheet Is Filled

Run:

```powershell
python -m src.summarize_error_review `
  --input outputs/error_analysis/distilbart_50_manual_review_template.csv `
  --output-json outputs/error_analysis/distilbart_50_manual_review_summary.json `
  --output-csv outputs/error_analysis/distilbart_50_manual_review_summary.csv
```

That gives counts and rates for each error type.

## CV Rule

Don't put a hallucination or factuality number on the CV until the filled review
sheet and summary files are in the repo.
