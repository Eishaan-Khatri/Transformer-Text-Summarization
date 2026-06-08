from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .create_error_review_template import ERROR_COLUMNS


TRUE_VALUES = {"1", "true", "yes", "y", "x"}


def is_flagged(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def summarize_review(input_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)
    counts = {column: 0 for column in ERROR_COLUMNS}
    any_error = 0
    for row in rows:
        row_has_error = False
        for column in ERROR_COLUMNS:
            if is_flagged(row.get(column)):
                counts[column] += 1
                row_has_error = True
        if row_has_error:
            any_error += 1
    return {
        "reviewed_examples": total,
        "error_counts": counts,
        "error_rates": {
            column: (counts[column] / total if total else 0.0)
            for column in ERROR_COLUMNS
        },
        "any_error_count": any_error,
        "any_error_rate": any_error / total if total else 0.0,
    }


def write_summary_csv(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "count", "rate"])
        writer.writeheader()
        for category, count in summary["error_counts"].items():
            writer.writerow(
                {
                    "category": category,
                    "count": count,
                    "rate": f"{summary['error_rates'][category]:.4f}",
                }
            )
        writer.writerow(
            {
                "category": "any_error",
                "count": summary["any_error_count"],
                "rate": f"{summary['any_error_rate']:.4f}",
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a manual error review CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", default="outputs/error_analysis/manual_review_summary.json")
    parser.add_argument("--output-csv", default="outputs/error_analysis/manual_review_summary.csv")
    args = parser.parse_args()
    summary = summarize_review(Path(args.input))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(summary, Path(args.output_csv))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
