from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .data import batch_records
from .metrics import compression_ratio


def _load_optional_dependencies():
    try:
        import evaluate
        from datasets import load_dataset
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Full benchmark requires torch, transformers, datasets, evaluate, and rouge_score. "
            "Install requirements.txt before running this script."
        ) from exc
    return evaluate, load_dataset, AutoModelForSeq2SeqLM, AutoTokenizer, torch


def run_model(
    model_name: str,
    split: str,
    sample_size: int,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    output_dir: Path,
) -> dict:
    evaluate, load_dataset, AutoModelForSeq2SeqLM, AutoTokenizer, torch = _load_optional_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split=f"{split}[:{sample_size}]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    rouge = evaluate.load("rouge")

    records = [
        {"id": row.get("id", str(index)), "article": row["article"], "reference_summary": row["highlights"]}
        for index, row in enumerate(dataset)
    ]

    predictions: list[str] = []
    references: list[str] = []
    rows: list[dict] = []
    start = time.perf_counter()
    for batch in batch_records(records, batch_size=batch_size):
        articles = [row["article"] for row in batch]
        tokenized = tokenizer(
            articles,
            max_length=max_input_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for record, prediction in zip(batch, decoded):
            predictions.append(prediction)
            references.append(record["reference_summary"])
            rows.append(
                {
                    "id": record["id"],
                    "prediction": prediction,
                    "reference_summary": record["reference_summary"],
                    "compression_ratio": compression_ratio(record["article"], prediction),
                }
            )
    elapsed_seconds = time.perf_counter() - start
    rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    avg_compression = sum(row["compression_ratio"] for row in rows) / len(rows)
    summary = {
        "model_name": model_name,
        "dataset": "ccdv/cnn_dailymail",
        "config": "3.0.0",
        "split": split,
        "sample_size": sample_size,
        "batch_size": batch_size,
        "device": device,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": len(rows) / elapsed_seconds if elapsed_seconds else 0.0,
        "average_compression_ratio": avg_compression,
        **rouge_scores,
    }

    safe_model_name = model_name.replace("/", "__")
    with (output_dir / f"{safe_model_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / f"{safe_model_name}_examples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction", "reference_summary", "compression_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN/DailyMail summarization benchmark.")
    parser.add_argument("--model", default="facebook/bart-large-cnn")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-dir", default="outputs/metrics/full_benchmark")
    args = parser.parse_args()
    summary = run_model(
        model_name=args.model,
        split=args.split,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

