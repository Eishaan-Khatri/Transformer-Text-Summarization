import unittest

import numpy as np

from src.scratch_transformer import (
    MiniSeq2SeqTransformer,
    MiniTransformerConfig,
    causal_mask,
    positional_encoding,
    scaled_dot_product_attention,
)


class ScratchTransformerTests(unittest.TestCase):
    def test_causal_mask_blocks_future_positions(self):
        mask = causal_mask(4)
        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])
        self.assertTrue(mask[3, 0])

    def test_positional_encoding_has_expected_shape(self):
        enc = positional_encoding(length=6, d_model=8)
        self.assertEqual(enc.shape, (6, 8))
        self.assertTrue(np.allclose(enc[0, 0::2], 0.0))
        self.assertTrue(np.allclose(enc[0, 1::2], 1.0))

    def test_attention_rows_sum_to_one_when_unmasked(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = query.copy()
        value = np.array([[[2.0, 0.0], [0.0, 2.0]]])
        output, weights = scaled_dot_product_attention(query, key, value)
        self.assertEqual(output.shape, (1, 2, 2))
        self.assertTrue(np.allclose(weights.sum(axis=-1), 1.0))

    def test_mini_transformer_forward_shape_is_vocab_sized(self):
        config = MiniTransformerConfig(vocab_size=32, d_model=16, num_heads=4, d_ff=32, max_length=12, seed=7)
        model = MiniSeq2SeqTransformer(config)
        source = np.array([[1, 4, 5, 2]])
        target = np.array([[1, 6, 2]])
        logits = model.forward(source, target)
        self.assertEqual(logits.shape, (1, 3, 32))


if __name__ == "__main__":
    unittest.main()
