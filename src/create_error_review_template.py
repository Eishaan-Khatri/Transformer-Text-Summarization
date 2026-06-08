from __future__ import annotations

import argparse
import csv
from pathlib import Path


ERROR_COLUMNS = [
    "missing_key_fact",
    "hallucination",
    "entity_error",
    "repetition",
    "over_compression",
]


def compact(text: str, max_chars: int = 900) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def load_examples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_review_template(
    examples_path: Path,
    output_path: Path,
    model_name: str,
    review_size: int,
) -> None:
    examples = load_examples(examples_path)[:review_size]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "example_id",
        "model_name",
        "article_excerpt",
        "reference_summary",
        "prediction",
        *ERROR_COLUMNS,
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(examples, start=1):
            writer.writerow(
                {
                    "review_id": index,
                    "example_id": row.get("id", ""),
                    "model_name": model_name,
                    "article_excerpt": compact(row.get("article", "")),
                    "reference_summary": compact(row.get("reference_summary", "")),
                    "prediction": compact(row.get("prediction", "")),
                    "missing_key_fact": "",
                    "hallucination": "",
                    "entity_error": "",
                    "repetition": "",
                    "over_compression": "",
                    "notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manual summarization error review CSV.")
    parser.add_argument("--examples", required=True, help="Model examples CSV from the benchmark output.")
    parser.add_argument("--output", default="outputs/error_analysis/manual_review_template.csv")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--review-size", type=int, default=50)
    args = parser.parse_args()
    examples_path = Path(args.examples)
    model_name = args.model_name or examples_path.stem.replace("_examples", "").replace("__", "/")
    write_review_template(
        examples_path=examples_path,
        output_path=Path(args.output),
        model_name=model_name,
        review_size=args.review_size,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
