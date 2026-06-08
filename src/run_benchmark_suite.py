from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .build_results import (
    ROOT,
    load_summaries,
    plot_metric_bars,
    run_lead_baselines,
    write_report,
    write_summary_table,
)
from .hf_benchmark import run_model


DEFAULT_MODELS = [
    "sshleifer/distilbart-cnn-6-6",
    "facebook/bart-large-cnn",
    "google/pegasus-cnn_dailymail",
]


def summary_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"{model_name.replace('/', '__')}_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a repeatable multi-model summarization benchmark.")
    parser.add_argument("--dataset", default="abisee/cnn_dailymail")
    parser.add_argument("--config", default="1.0.0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-dir", default="outputs/metrics/full_benchmark")
    parser.add_argument("--table-output", default="outputs/tables/benchmark_summary.csv")
    parser.add_argument("--figures-output", default="outputs/figures")
    parser.add_argument("--report-output", default="reports/final_report.md")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "offset": args.offset,
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "models": args.models,
        "run_baselines": args.run_baselines,
        "started_at_unix": time.time(),
        "completed_models": [],
        "skipped_models": [],
    }

    if args.run_baselines:
        run_lead_baselines(
            output_dir=output_dir,
            dataset_name=args.dataset,
            config_name=args.config,
            split=args.split,
            sample_size=args.sample_size,
            offset=args.offset,
        )

    for model_name in args.models:
        if args.skip_existing and summary_path(output_dir, model_name).exists():
            manifest["skipped_models"].append(model_name)
            continue
        summary = run_model(
            model_name=model_name,
            dataset_name=args.dataset,
            config_name=args.config,
            split=args.split,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            output_dir=output_dir,
            offset=args.offset,
        )
        manifest["completed_models"].append(
            {
                "model_name": model_name,
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "examples_per_second": summary.get("examples_per_second"),
            }
        )

    rows = load_summaries(output_dir)
    write_summary_table(rows, ROOT / args.table_output)
    plot_metric_bars(rows, ROOT / args.figures_output)
    write_report(rows, ROOT / args.report_output)
    manifest["finished_at_unix"] = time.time()
    manifest["summary_count"] = len(rows)
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
