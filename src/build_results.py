from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .hf_benchmark import load_records_from_viewer
from .lead_baseline import lead_summary
from .metrics import compression_ratio


ROOT = Path(__file__).resolve().parents[1]


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_").lower()


def display_name(name: str) -> str:
    names = {
        "lead_1_baseline": "Lead-1",
        "lead_2_baseline": "Lead-2",
        "lead_3_baseline": "Lead-3",
        "sshleifer/distilbart-cnn-6-6": "DistilBART CNN",
    }
    return names.get(name, name)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_lead_baselines(
    output_dir: Path,
    dataset_name: str,
    config_name: str,
    split: str,
    sample_size: int,
    offset: int,
) -> list[dict]:
    import evaluate
    import requests

    records = load_records_from_viewer(
        requests_module=requests,
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        sample_size=sample_size,
        offset=offset,
    )
    rouge = evaluate.load("rouge")
    summaries: list[dict] = []

    for sentence_count in [1, 2, 3]:
        model_name = f"lead_{sentence_count}_baseline"
        start = time.perf_counter()
        predictions = [lead_summary(row["article"], sentences=sentence_count) for row in records]
        elapsed = time.perf_counter() - start
        references = [row["reference_summary"] for row in records]
        rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        compression_values = [
            compression_ratio(row["article"], prediction)
            for row, prediction in zip(records, predictions)
        ]
        summary = {
            "model_name": model_name,
            "dataset": dataset_name,
            "config": config_name,
            "split": split,
            "sample_size": len(records),
            "offset": offset,
            "batch_size": 0,
            "device": "cpu",
            "elapsed_seconds": elapsed,
            "examples_per_second": len(records) / elapsed if elapsed else 0.0,
            "average_compression_ratio": sum(compression_values) / len(compression_values),
            **rouge_scores,
        }
        example_rows = [
            {
                "id": row["id"],
                "prediction": prediction,
                "reference_summary": row["reference_summary"],
                "compression_ratio": value,
            }
            for row, prediction, value in zip(records, predictions, compression_values)
        ]
        with (output_dir / f"{model_name}_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        write_rows(output_dir / f"{model_name}_examples.csv", example_rows)
        summaries.append(summary)
    return summaries


def load_summaries(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_dir.glob("*_summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def write_summary_table(rows: list[dict], output_path: Path) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "model": row["model_name"],
                "dataset": row["dataset"],
                "sample_size": row["sample_size"],
                "device": row.get("device", ""),
                "batch_size": row.get("batch_size", ""),
                "rouge1": f"{row.get('rouge1', 0):.4f}",
                "rouge2": f"{row.get('rouge2', 0):.4f}",
                "rougeL": f"{row.get('rougeL', 0):.4f}",
                "rougeLsum": f"{row.get('rougeLsum', 0):.4f}",
                "compression_ratio": f"{row.get('average_compression_ratio', 0):.4f}",
                "examples_per_second": f"{row.get('examples_per_second', 0):.4f}",
                "elapsed_seconds": f"{row.get('elapsed_seconds', 0):.2f}",
            }
        )
    write_rows(output_path, table_rows)


def plot_metric_bars(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [display_name(row["model_name"]) for row in rows]

    rouge_metrics = ["rouge1", "rouge2", "rougeL"]
    x = range(len(labels))
    width = 0.24
    plt.figure(figsize=(10, 5))
    for idx, metric in enumerate(rouge_metrics):
        values = [row.get(metric, 0) for row in rows]
        positions = [value + (idx - 1) * width for value in x]
        plt.bar(positions, values, width=width, label=metric)
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylabel("ROUGE F1")
    plt.title("Summarization quality on CNN/DailyMail sample")
    plt.ylim(0, max(max(row.get(metric, 0) for row in rows) for metric in rouge_metrics) + 0.08)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rouge_comparison.svg")
    plt.savefig(output_dir / "rouge_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, [row.get("average_compression_ratio", 0) for row in rows], color="#4C78A8")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Summary tokens / article tokens")
    plt.title("Compression ratio")
    plt.tight_layout()
    plt.savefig(output_dir / "compression_ratio.svg")
    plt.savefig(output_dir / "compression_ratio.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, [row.get("examples_per_second", 0) for row in rows], color="#F58518")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Examples per second")
    plt.yscale("log")
    plt.title("CPU throughput, log scale")
    plt.tight_layout()
    plt.savefig(output_dir / "throughput.svg")
    plt.savefig(output_dir / "throughput.png", dpi=160)
    plt.close()


def markdown_table(rows: list[dict]) -> str:
    header = "| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Ex/sec |"
    divider = "|---|---:|---:|---:|---:|---:|"
    body = [
        (
            f"| {display_name(row['model_name'])} | {row.get('rouge1', 0):.4f} | {row.get('rouge2', 0):.4f} | "
            f"{row.get('rougeL', 0):.4f} | {row.get('average_compression_ratio', 0):.4f} | "
            f"{row.get('examples_per_second', 0):.4f} |"
        )
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_report(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = rows[0]["dataset"] if rows else ""
    sample_size = rows[0]["sample_size"] if rows else ""
    lines = [
        "# Summarization Results",
        "",
        f"I ran this on `{dataset}` with `{sample_size}` test examples. This is a small CPU run, so I use it as a checked experiment, not as a final leaderboard number.",
        "",
        "## Metrics",
        "",
        markdown_table(rows),
        "",
        "## Charts",
        "",
        "![ROUGE comparison](../outputs/figures/rouge_comparison.png)",
        "",
        "![Compression ratio](../outputs/figures/compression_ratio.png)",
        "",
        "![CPU throughput](../outputs/figures/throughput.png)",
        "",
        "## What I take from this run",
        "",
        "- DistilBART gives the best ROUGE numbers in this small run, but it is slow on CPU.",
        "- Lead baselines are useful because CNN/DailyMail articles often put important facts early.",
        "- Baseline throughput is simple sentence slicing, so it should not be read as model inference speed.",
        "- Compression ratio matters separately from ROUGE; a shorter summary is not automatically better.",
        "- I would not use this as a final claim until running a larger sample or the full test split.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build result tables and charts.")
    parser.add_argument("--dataset", default="abisee/cnn_dailymail")
    parser.add_argument("--config", default="1.0.0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--output-dir", default="outputs/metrics/full_benchmark")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    if args.run_baselines:
        run_lead_baselines(
            output_dir=output_dir,
            dataset_name=args.dataset,
            config_name=args.config,
            split=args.split,
            sample_size=args.sample_size,
            offset=args.offset,
        )
    rows = load_summaries(output_dir)
    if not rows:
        raise RuntimeError(f"No summary JSON files found in {output_dir}.")
    write_summary_table(rows, ROOT / "outputs" / "tables" / "benchmark_summary.csv")
    plot_metric_bars(rows, ROOT / "outputs" / "figures")
    write_report(rows, ROOT / "reports" / "final_report.md")
    print(json.dumps({"models": [row["model_name"] for row in rows], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
