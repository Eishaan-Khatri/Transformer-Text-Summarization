from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, predicted_total: int, reference_total: int) -> float:
    if overlap == 0 or predicted_total == 0 or reference_total == 0:
        return 0.0
    precision = overlap / predicted_total
    recall = overlap / reference_total
    return 2 * precision * recall / (precision + recall)


def rouge_n_f1(prediction: str, reference: str, n: int = 1) -> float:
    predicted = _ngrams(tokenize(prediction), n)
    expected = _ngrams(tokenize(reference), n)
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    return _f1(overlap, sum(predicted.values()), sum(expected.values()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str, beta: float = 1.2) -> float:
    predicted_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(predicted_tokens, reference_tokens)
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(reference_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / ((beta_squared * precision) + recall)


def compression_ratio(article: str, summary: str) -> float:
    article_tokens = tokenize(article)
    summary_tokens = tokenize(summary)
    if not article_tokens:
        return 0.0
    return len(summary_tokens) / len(article_tokens)


def compute_summary_metrics(article: str, prediction: str, reference: str) -> dict[str, float]:
    return {
        "rouge1": rouge_n_f1(prediction, reference, n=1),
        "rouge2": rouge_n_f1(prediction, reference, n=2),
        "rougeL": rouge_l_f1(prediction, reference),
        "compression_ratio": compression_ratio(article, prediction),
    }

