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

To test that properly, I would add:

1. a small query set,
2. an index built from titles only,
3. an index built from titles plus summaries,
4. Recall@K or MRR comparison,
5. manual error analysis where summaries help or hurt.
