import math
import unittest

from src.metrics import compression_ratio, rouge_l_f1, rouge_n_f1


class MetricTests(unittest.TestCase):
    def test_rouge_1_rewards_unigram_overlap(self):
        score = rouge_n_f1("cat sat on mat", "cat slept on mat", n=1)
        self.assertAlmostEqual(score, 0.75, places=6)

    def test_rouge_2_uses_contiguous_bigrams(self):
        score = rouge_n_f1("cat sat on mat", "cat slept on mat", n=2)
        self.assertAlmostEqual(score, 1 / 3, places=6)

    def test_rouge_l_uses_longest_common_subsequence(self):
        score = rouge_l_f1("alpha beta gamma delta", "alpha gamma delta", beta=1.0)
        self.assertAlmostEqual(score, 6 / 7, places=6)

    def test_compression_ratio_uses_output_over_input_tokens(self):
        ratio = compression_ratio("one two three four", "one two")
        self.assertTrue(math.isclose(ratio, 0.5))

    def test_compression_ratio_returns_zero_for_empty_output(self):
        self.assertEqual(compression_ratio("one two three", ""), 0.0)


if __name__ == "__main__":
    unittest.main()
