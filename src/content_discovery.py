from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from .metrics import tokenize


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def content_terms(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2]


def top_tags(text: str, limit: int = 8) -> list[str]:
    return [term for term, _ in Counter(content_terms(text)).most_common(limit)]


def tfidf_vectors(rows: list[dict[str, str]], text_column: str) -> list[Counter[str]]:
    documents = [content_terms(row.get(text_column, "")) for row in rows]
    document_frequency: Counter[str] = Counter()
    for terms in documents:
        document_frequency.update(set(terms))
    total = len(documents)
    vectors: list[Counter[str]] = []
    for terms in documents:
        counts = Counter(terms)
        weighted: Counter[str] = Counter()
        for term, count in counts.items():
            idf = math.log((1 + total) / (1 + document_frequency[term])) + 1
            weighted[term] = count * idf
        vectors.append(weighted)
    return vectors


def cosine(left: Counter[str], right: Counter[str]) -> float:
    shared = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def nearest_neighbors(rows: list[dict[str, str]], vectors: list[Counter[str]], top_k: int) -> list[dict]:
    output: list[dict] = []
    for index, row in enumerate(rows):
        scored = []
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            scored.append((cosine(vectors[index], vectors[other_index]), other_index, other))
        scored.sort(reverse=True, key=lambda item: item[0])
        for rank, (score, _, other) in enumerate(scored[:top_k], start=1):
            output.append(
                {
                    "query_id": row.get("id", index),
                    "neighbor_id": other.get("id", ""),
                    "rank": rank,
                    "score": f"{score:.6f}",
                    "query_tags": ", ".join(top_tags(row.get("prediction", ""))),
                    "neighbor_tags": ", ".join(top_tags(other.get("prediction", ""))),
                }
            )
    return output


def evaluate_label_retrieval(
    rows: list[dict[str, str]],
    neighbors: list[dict],
    label_column: str,
    top_k: int,
) -> dict:
    labels = {str(row.get("id", index)): row.get(label_column, "") for index, row in enumerate(rows)}
    if not any(labels.values()):
        return {"available": False, "reason": f"label column '{label_column}' is empty or missing"}

    query_hits: dict[str, bool] = {}
    reciprocal_ranks: dict[str, float] = {}
    for item in neighbors:
        query_id = str(item["query_id"])
        if int(item["rank"]) > top_k:
            continue
        if query_id in query_hits and query_hits[query_id]:
            continue
        query_label = labels.get(query_id, "")
        neighbor_label = labels.get(str(item["neighbor_id"]), "")
        is_match = bool(query_label and query_label == neighbor_label)
        query_hits[query_id] = query_hits.get(query_id, False) or is_match
        if is_match and query_id not in reciprocal_ranks:
            reciprocal_ranks[query_id] = 1 / int(item["rank"])

    total = len(rows)
    recall_at_k = sum(1 for hit in query_hits.values() if hit) / total if total else 0.0
    mrr = sum(reciprocal_ranks.values()) / total if total else 0.0
    return {
        "available": True,
        "label_column": label_column,
        "top_k": top_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a summary-based content-discovery bridge.")
    parser.add_argument("--examples", required=True, help="Model examples CSV with prediction column.")
    parser.add_argument("--text-column", default="prediction")
    parser.add_argument("--label-column", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/content_discovery")
    args = parser.parse_args()

    rows = load_rows(Path(args.examples))
    vectors = tfidf_vectors(rows, args.text_column)
    item_rows = [
        {
            "id": row.get("id", index),
            "summary": row.get(args.text_column, ""),
            "tags": ", ".join(top_tags(row.get(args.text_column, ""))),
        }
        for index, row in enumerate(rows)
    ]
    neighbors = nearest_neighbors(rows, vectors, args.top_k)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "summary_tags.csv", item_rows, ["id", "summary", "tags"])
    write_csv(
        output_dir / "summary_neighbors.csv",
        neighbors,
        ["query_id", "neighbor_id", "rank", "score", "query_tags", "neighbor_tags"],
    )

    label_eval = (
        evaluate_label_retrieval(rows, neighbors, args.label_column, args.top_k)
        if args.label_column
        else {"available": False, "reason": "no label column supplied"}
    )
    summary = {
        "examples": len(rows),
        "embedding_type": "tfidf_over_generated_summaries",
        "top_k": args.top_k,
        "label_evaluation": label_eval,
        "outputs": {
            "tags": str(output_dir / "summary_tags.csv"),
            "neighbors": str(output_dir / "summary_neighbors.csv"),
        },
    }
    (output_dir / "retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
