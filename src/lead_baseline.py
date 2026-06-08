from __future__ import annotations

import re


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split((text or "").strip()) if part.strip()]


def lead_summary(article: str, sentences: int = 3, max_words: int | None = None) -> str:
    if sentences <= 0:
        raise ValueError("sentences must be positive")
    summary = " ".join(split_sentences(article)[:sentences]).strip()
    if max_words is not None and max_words > 0:
        words = summary.split()
        summary = " ".join(words[:max_words])
    return summary


def summarize_batch(records: list[dict], sentences: int = 3, max_words: int | None = None) -> list[str]:
    return [lead_summary(record["article"], sentences=sentences, max_words=max_words) for record in records]

