import csv
import tempfile
import unittest
from pathlib import Path

from src.create_error_review_template import write_review_template
from src.summarize_error_review import summarize_review


class ErrorReviewTests(unittest.TestCase):
    def test_review_template_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            examples_path = tmp_path / "examples.csv"
            with examples_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["id", "article", "prediction", "reference_summary", "compression_ratio"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "1",
                        "article": "Long article text",
                        "prediction": "Short prediction",
                        "reference_summary": "Reference",
                        "compression_ratio": "0.2",
                    }
                )

            review_path = tmp_path / "review.csv"
            write_review_template(examples_path, review_path, "model", review_size=1)
            with review_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            rows[0]["hallucination"] = "yes"
            with review_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            summary = summarize_review(review_path)
            self.assertEqual(summary["reviewed_examples"], 1)
            self.assertEqual(summary["error_counts"]["hallucination"], 1)


if __name__ == "__main__":
    unittest.main()
