from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import load_jsonl, write_jsonl
from .lead_baseline import lead_summary
from .metrics import compute_summary_metrics
from .scratch_transformer import MiniSeq2SeqTransformer, MiniTransformerConfig


ROOT = Path(__file__).resolve().parents[1]


def run_demo() -> dict:
    sample_path = ROOT / "data" / "sample" / "news_examples.jsonl"
    records = load_jsonl(sample_path)
    example_rows: list[dict] = []
    metric_rows: list[dict] = []

    for record in records:
        prediction = lead_summary(record["article"], sentences=2, max_words=70)
        metrics = compute_summary_metrics(record["article"], prediction, record["reference_summary"])
        row = {
            "id": record["id"],
            "model": "lead_2_baseline",
            "prediction": prediction,
            "reference_summary": record["reference_summary"],
            **metrics,
        }
        example_rows.append(row)
        metric_rows.append({key: row[key] for key in ["id", "model", "rouge1", "rouge2", "rougeL", "compression_ratio"]})

    config = MiniTransformerConfig(vocab_size=64, d_model=32, num_heads=4, d_ff=64, max_length=16, seed=11)
    scratch_model = MiniSeq2SeqTransformer(config)
    logits = scratch_model.forward(np.array([[1, 5, 6, 7, 2]]), np.array([[1, 8, 9, 2]]))

    output_examples = ROOT / "outputs" / "examples" / "sample_summaries.jsonl"
    output_metrics_json = ROOT / "outputs" / "metrics" / "lightweight_demo_metrics.json"
    output_metrics_csv = ROOT / "outputs" / "metrics" / "lightweight_demo_metrics.csv"
    write_jsonl(output_examples, example_rows)
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "records": len(records),
        "model": "lead_2_baseline",
        "mean_rouge1": sum(row["rouge1"] for row in metric_rows) / len(metric_rows),
        "mean_rouge2": sum(row["rouge2"] for row in metric_rows) / len(metric_rows),
        "mean_rougeL": sum(row["rougeL"] for row in metric_rows) / len(metric_rows),
        "mean_compression_ratio": sum(row["compression_ratio"] for row in metric_rows) / len(metric_rows),
        "scratch_transformer_logits_shape": list(logits.shape),
        "status": "local demo only; benchmark results come from src.hf_benchmark and src.build_results",
    }
    with output_metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with output_metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    final_report = ROOT / "reports" / "final_report.md"
    final_report.write_text(
        "\n".join(
            [
                "# Text Summarization with Transformers - Local Demo Report",
                "",
                "This report is generated from the local demo, not the CNN/DailyMail benchmark.",
                "",
                "## Local Demo Metrics",
                "",
                f"- Records evaluated: {summary['records']}",
                f"- Baseline: {summary['model']}",
                f"- Mean ROUGE-1: {summary['mean_rouge1']:.4f}",
                f"- Mean ROUGE-2: {summary['mean_rouge2']:.4f}",
                f"- Mean ROUGE-L: {summary['mean_rougeL']:.4f}",
                f"- Mean compression ratio: {summary['mean_compression_ratio']:.4f}",
                f"- Scratch Transformer logits shape: {summary['scratch_transformer_logits_shape']}",
                "",
                "## Interpretation",
                "",
                "The run checks that the data, metric, baseline, and scratch Transformer components execute locally.",
                "For benchmark numbers, use `src.hf_benchmark` and `src.build_results`.",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
