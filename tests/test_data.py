import unittest

from src.data import batch_records, normalize_record


class DataTests(unittest.TestCase):
    def test_normalize_record_requires_article_and_summary(self):
        record = normalize_record({"article": " Body text ", "highlights": " Summary "})
        self.assertEqual(record["article"], "Body text")
        self.assertEqual(record["reference_summary"], "Summary")

    def test_batch_records_keeps_order_and_batch_size(self):
        records = [{"id": str(i)} for i in range(5)]
        batches = list(batch_records(records, batch_size=2))
        self.assertEqual([[row["id"] for row in batch] for batch in batches], [["0", "1"], ["2", "3"], ["4"]])

    def test_batch_records_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            list(batch_records([{"id": "1"}], batch_size=0))


if __name__ == "__main__":
    unittest.main()
