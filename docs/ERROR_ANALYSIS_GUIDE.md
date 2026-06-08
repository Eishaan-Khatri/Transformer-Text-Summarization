# Error Analysis Guide

ROUGE is useful, but it can miss obvious problems. A summary can share words
with the reference and still be wrong.

That is why this project now has a manual review sheet.

## Review File

```text
outputs/error_analysis/distilbart_50_manual_review_template.csv
```

Each row has:

- article excerpt,
- reference summary,
- model prediction,
- error flags,
- notes.

## Error Categories

Use `yes` when the error is present. Leave it blank when it is not.

| Category | What it means |
|---|---|
| missing_key_fact | The summary skips a main fact from the article/reference |
| hallucination | The summary adds a fact that is not supported by the article |
| entity_error | The summary gets a person, place, group, date, or number wrong |
| repetition | The summary repeats the same point in a distracting way |
| over_compression | The summary is too short and loses the useful part |

## How To Review

For each row:

1. Read the reference summary first.
2. Read the model prediction.
3. Check the article excerpt only when something looks missing or suspicious.
4. Mark one or more error categories with `yes`.
5. Add a short note when the error is not obvious.

## Summarize The Review

After filling the sheet:

```powershell
python -m src.summarize_error_review `
  --input outputs/error_analysis/distilbart_50_manual_review_template.csv `
  --output-json outputs/error_analysis/distilbart_50_manual_review_summary.json `
  --output-csv outputs/error_analysis/distilbart_50_manual_review_summary.csv
```

The output gives:

- count per error type,
- rate per error type,
- any-error rate.

## Resume Rule

Do not put a factuality or hallucination number on the CV unless the filled
review sheet and summary files are committed in the repo.
