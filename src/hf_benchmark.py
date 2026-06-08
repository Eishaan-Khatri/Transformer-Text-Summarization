from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.parse import quote

from .data import batch_records
from .metrics import compression_ratio, tokenize


def _load_optional_dependencies():
    try:
        import evaluate
        import requests
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Full benchmark requires torch, transformers, datasets, evaluate, rouge_score, and requests. "
            "Install requirements.txt before running this script."
        ) from exc
    return evaluate, requests, AutoModelForSeq2SeqLM, AutoTokenizer, torch


def records_from_viewer_payload(payload: dict) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for item in payload.get("rows", []):
        row = item.get("row", {})
        article = str(row.get("article", "")).strip()
        reference = str(row.get("highlights", row.get("summary", ""))).strip()
        if not article or not reference:
            continue
        records.append(
            {
                "id": str(row.get("id", item.get("row_idx", len(records)))),
                "article": article,
                "reference_summary": reference,
            }
        )
    return records


def load_records_from_viewer(
    requests_module,
    dataset_name: str,
    config_name: str,
    split: str,
    sample_size: int,
    offset: int = 0,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    page_size = min(100, sample_size)
    encoded_dataset = quote(dataset_name, safe="/")
    encoded_config = quote(config_name, safe="")
    encoded_split = quote(split, safe="")
    while len(records) < sample_size:
        remaining = sample_size - len(records)
        length = min(page_size, remaining)
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={encoded_dataset}&config={encoded_config}&split={encoded_split}"
            f"&offset={offset + len(records)}&length={length}"
        )
        response = requests_module.get(url, timeout=60)
        response.raise_for_status()
        page_records = records_from_viewer_payload(response.json())
        if not page_records:
            break
        records.extend(page_records)
    return records


def run_model(
    model_name: str,
    dataset_name: str,
    config_name: str,
    split: str,
    sample_size: int,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    output_dir: Path,
    offset: int = 0,
) -> dict:
    evaluate, requests, AutoModelForSeq2SeqLM, AutoTokenizer, torch = _load_optional_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records_from_viewer(
        requests_module=requests,
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        sample_size=sample_size,
        offset=offset,
    )
    if not records:
        raise RuntimeError(f"No rows loaded from {dataset_name}/{config_name}/{split}.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    rouge = evaluate.load("rouge")

    predictions: list[str] = []
    references: list[str] = []
    rows: list[dict] = []
    start = time.perf_counter()
    batch_latencies: list[float] = []
    for batch in batch_records(records, batch_size=batch_size):
        articles = [row["article"] for row in batch]
        tokenized = tokenizer(
            articles,
            max_length=max_input_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        batch_start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        batch_elapsed = time.perf_counter() - batch_start
        per_example_latency = batch_elapsed / len(batch)
        batch_latencies.extend([per_example_latency] * len(batch))
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for record, prediction in zip(batch, decoded):
            predictions.append(prediction)
            references.append(record["reference_summary"])
            rows.append(
                {
                    "id": record["id"],
                    "article": record["article"],
                    "prediction": prediction,
                    "reference_summary": record["reference_summary"],
                    "compression_ratio": compression_ratio(record["article"], prediction),
                    "article_tokens": len(tokenize(record["article"])),
                    "prediction_tokens": len(tokenize(prediction)),
                    "latency_seconds": per_example_latency,
                }
            )
    elapsed_seconds = time.perf_counter() - start
    rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    avg_compression = sum(row["compression_ratio"] for row in rows) / len(rows)
    summary = {
        "model_name": model_name,
        "dataset": dataset_name,
        "config": config_name,
        "split": split,
        "sample_size": len(records),
        "offset": offset,
        "batch_size": batch_size,
        "device": device,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": len(rows) / elapsed_seconds if elapsed_seconds else 0.0,
        "average_latency_seconds": sum(batch_latencies) / len(batch_latencies) if batch_latencies else 0.0,
        "average_compression_ratio": avg_compression,
        **rouge_scores,
    }

    safe_model_name = model_name.replace("/", "__")
    with (output_dir / f"{safe_model_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / f"{safe_model_name}_examples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "article",
                "prediction",
                "reference_summary",
                "compression_ratio",
                "article_tokens",
                "prediction_tokens",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN/DailyMail summarization benchmark.")
    parser.add_argument("--model", default="sshleifer/distilbart-cnn-6-6")
    parser.add_argument("--dataset", default="abisee/cnn_dailymail")
    parser.add_argument("--config", default="1.0.0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-dir", default="outputs/metrics/full_benchmark")
    args = parser.parse_args()
    summary = run_model(
        model_name=args.model,
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        output_dir=Path(args.output_dir),
        offset=args.offset,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
