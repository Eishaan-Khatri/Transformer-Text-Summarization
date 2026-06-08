from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def normalize_record(record: dict) -> dict[str, str]:
    article = str(record.get("article", "")).strip()
    reference = str(record.get("reference_summary", record.get("highlights", record.get("summary", "")))).strip()
    if not article:
        raise ValueError("record is missing a non-empty article")
    if not reference:
        raise ValueError("record is missing a non-empty reference summary")
    normalized = {
        "id": str(record.get("id", "")),
        "article": article,
        "reference_summary": reference,
    }
    return normalized


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(normalize_record(json.loads(line)))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def batch_records(records: Iterable[dict], batch_size: int) -> Iterator[list[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[dict] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

