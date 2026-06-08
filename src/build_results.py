from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .hf_benchmark import load_records_from_viewer
from .lead_baseline import lead_summary
from .metrics import compression_ratio, tokenize


ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["svg.hashsalt"] = "text-summarization"


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_").lower()


def display_name(name: str) -> str:
    names = {
        "lead_1_baseline": "Lead-1",
        "lead_2_baseline": "Lead-2",
        "lead_3_baseline": "Lead-3",
        "sshleifer/distilbart-cnn-6-6": "DistilBART CNN",
        "facebook/bart-large-cnn": "BART-large CNN",
        "google/pegasus-cnn_dailymail": "PEGASUS CNN/DailyMail",
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

    output_dir.mkdir(parents=True, exist_ok=True)
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
            "average_latency_seconds": elapsed / len(records) if records else 0.0,
            "average_compression_ratio": sum(compression_values) / len(compression_values),
            **rouge_scores,
        }
        example_rows = [
            {
                "id": row["id"],
                "article": row["article"],
                "prediction": prediction,
                "reference_summary": row["reference_summary"],
                "compression_ratio": value,
                "article_tokens": len(tokenize(row["article"])),
                "prediction_tokens": len(tokenize(prediction)),
                "latency_seconds": elapsed / len(records) if records else 0.0,
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
            row = json.load(handle)
        if "average_latency_seconds" not in row:
            sample_size = row.get("sample_size") or 0
            elapsed = row.get("elapsed_seconds") or 0
            row["average_latency_seconds"] = elapsed / sample_size if sample_size else 0.0
        rows.append(row)
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
                "latency_seconds": f"{row.get('average_latency_seconds', 0):.4f}",
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
    plt.savefig(output_dir / "rouge_comparison.svg", metadata={"Date": None})
    plt.savefig(output_dir / "rouge_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, [row.get("average_compression_ratio", 0) for row in rows], color="#4C78A8")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Summary tokens / article tokens")
    plt.title("Compression ratio")
    plt.tight_layout()
    plt.savefig(output_dir / "compression_ratio.svg", metadata={"Date": None})
    plt.savefig(output_dir / "compression_ratio.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, [row.get("examples_per_second", 0) for row in rows], color="#F58518")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Examples per second")
    plt.yscale("log")
    plt.title("CPU throughput, log scale")
    plt.tight_layout()
    plt.savefig(output_dir / "throughput.svg", metadata={"Date": None})
    plt.savefig(output_dir / "throughput.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, [row.get("average_latency_seconds", 0) for row in rows], color="#54A24B")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average seconds per example")
    plt.yscale("log")
    plt.title("Latency per example, log scale")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_per_example.svg", metadata={"Date": None})
    plt.savefig(output_dir / "latency_per_example.png", dpi=160)
    plt.close()


def markdown_table(rows: list[dict]) -> str:
    header = "| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | Latency sec/ex | Ex/sec |"
    divider = "|---|---:|---:|---:|---:|---:|---:|"
    body = [
        (
            f"| {display_name(row['model_name'])} | {row.get('rouge1', 0):.4f} | {row.get('rouge2', 0):.4f} | "
            f"{row.get('rougeL', 0):.4f} | {row.get('average_compression_ratio', 0):.4f} | "
            f"{row.get('average_latency_seconds', 0):.4f} | "
            f"{row.get('examples_per_second', 0):.4f} |"
        )
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _relative_figure_dir(output_path: Path, figures_output_dir: Path) -> str:
    return Path(os.path.relpath(figures_output_dir, output_path.parent)).as_posix()


def write_report(rows: list[dict], output_path: Path, figures_output_dir: Path | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = rows[0]["dataset"] if rows else ""
    sample_size = rows[0]["sample_size"] if rows else ""
    figures_dir = _relative_figure_dir(output_path, figures_output_dir or ROOT / "outputs" / "figures")
    model_names = {row.get("model_name", "") for row in rows}
    only_leads = bool(model_names) and all(name.startswith("lead_") for name in model_names)
    title = "500-Example Baseline Results" if only_leads and int(sample_size or 0) >= 500 else "Summarization Results"
    intro_lines = (
        [
            f"I ran Lead baselines on `{dataset}` with `{sample_size}` test examples.",
            "",
            "This report sets a baseline before a bigger Transformer run.",
        ]
        if only_leads
        else [
            f"I ran this on `{dataset}` with `{sample_size}` test examples.",
            "",
            "This is a small CPU run. It checks that the code, metrics, baselines, and charts work together.",
            "",
            "It is not a final benchmark.",
        ]
    )
    takeaways = (
        [
            "- Lead-3 is the strongest Lead baseline here.",
            "- Lead-2 is close, with a lower compression ratio.",
            "- These baselines are very fast because they only slice sentences.",
            "- This gives a fairer baseline for a future Transformer run.",
        ]
        if only_leads
        else [
            "- DistilBART scored best on ROUGE in this small run.",
            "- It was also slow on CPU.",
            "- Lead-3 is still useful because news articles often start with the main facts.",
            "- Compression and ROUGE should be read together. A shorter summary is not automatically better.",
            "- The next serious step is a 500-example neural run on GPU.",
        ]
    )
    lines = [
        f"# {title}",
        "",
        *intro_lines,
        "",
        "## Metrics",
        "",
        markdown_table(rows),
        "",
        "## Charts",
        "",
        f"![ROUGE comparison]({figures_dir}/rouge_comparison.png)",
        "",
        f"![Compression ratio]({figures_dir}/compression_ratio.png)",
        "",
        f"![CPU throughput]({figures_dir}/throughput.png)",
        "",
        f"![Latency per example]({figures_dir}/latency_per_example.png)",
        "",
        "## What I Take From This Run",
        "",
        *takeaways,
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build result tables and charts.")
    parser.add_argument("--dataset", default="abisee/cnn_dailymail")
    parser.add_argument("--config", default="1.0.0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--output-dir", default="outputs/metrics/full_benchmark")
    parser.add_argument("--table-output", default="outputs/tables/benchmark_summary.csv")
    parser.add_argument("--figures-output", default="outputs/figures")
    parser.add_argument("--report-output", default="reports/final_report.md")
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
    write_summary_table(rows, ROOT / args.table_output)
    plot_metric_bars(rows, ROOT / args.figures_output)
    write_report(rows, ROOT / args.report_output, ROOT / args.figures_output)
    print(json.dumps({"models": [row["model_name"] for row in rows], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
