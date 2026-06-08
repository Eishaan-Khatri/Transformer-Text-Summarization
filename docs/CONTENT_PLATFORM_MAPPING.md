# Content Platform Mapping

This project connects to content platforms because many platforms need short
representations of long text.

Examples:

- article cards,
- story previews,
- search snippets,
- cold-start metadata,
- editorial review queues.

The current project only proves summarization and measurement. It does not prove
that summaries improve search or recommendations.

The repo now has a small bridge script:

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

This is still not proof that search improves. It creates the structure needed
for that test.

To test search properly, I would add:

1. a small query set,
2. an index built from titles only,
3. an index built from titles plus summaries,
4. Recall@K or MRR comparison,
5. manual error analysis where summaries help or hurt.

If category or genre labels are available, the script can compute labelled
retrieval metrics with `--label-column`.
