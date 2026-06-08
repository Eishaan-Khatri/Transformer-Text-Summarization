import unittest

from src.hf_benchmark import records_from_viewer_payload


class DatasetViewerTests(unittest.TestCase):
    def test_records_from_viewer_payload_keeps_article_highlights_and_id(self):
        payload = {
            "rows": [
                {
                    "row_idx": 3,
                    "row": {
                        "id": "abc",
                        "article": "Article body",
                        "highlights": "Short summary",
                    },
                }
            ]
        }
        records = records_from_viewer_payload(payload)
        self.assertEqual(
            records,
            [{"id": "abc", "article": "Article body", "reference_summary": "Short summary"}],
        )


if __name__ == "__main__":
    unittest.main()
