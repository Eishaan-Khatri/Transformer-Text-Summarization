import unittest
from collections import Counter

from src.content_discovery import cosine, evaluate_label_retrieval, nearest_neighbors, top_tags


class ContentDiscoveryTests(unittest.TestCase):
    def test_top_tags_removes_common_words(self):
        tags = top_tags("The city city opened a dashboard for bus delays")
        self.assertEqual(tags[0], "city")
        self.assertNotIn("the", tags)
        self.assertNotIn("for", tags)

    def test_cosine_scores_identical_vectors_as_one(self):
        self.assertAlmostEqual(cosine(Counter({"news": 2}), Counter({"news": 2})), 1.0)

    def test_label_retrieval_computes_recall_and_mrr(self):
        rows = [
            {"id": "a", "prediction": "football match", "category": "sports"},
            {"id": "b", "prediction": "football team", "category": "sports"},
            {"id": "c", "prediction": "budget vote", "category": "politics"},
        ]
        vectors = [Counter({"football": 1}), Counter({"football": 1}), Counter({"budget": 1})]
        neighbors = nearest_neighbors(rows, vectors, top_k=1)
        summary = evaluate_label_retrieval(rows, neighbors, "category", top_k=1)
        self.assertTrue(summary["available"])
        self.assertGreater(summary["recall_at_k"], 0)
        self.assertGreater(summary["mrr"], 0)


if __name__ == "__main__":
    unittest.main()
