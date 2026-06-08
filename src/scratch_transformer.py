from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def causal_mask(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    return np.tril(np.ones((length, length), dtype=bool))


def positional_encoding(length: int, d_model: int) -> np.ndarray:
    if length <= 0 or d_model <= 0:
        raise ValueError("length and d_model must be positive")
    positions = np.arange(length)[:, None]
    dimensions = np.arange(0, d_model, 2)
    angles = positions / np.power(10000, dimensions / d_model)
    encoding = np.zeros((length, d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(query.shape[-1])
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / scale
    if mask is not None:
        broadcast_mask = mask
        while broadcast_mask.ndim < scores.ndim:
            broadcast_mask = np.expand_dims(broadcast_mask, axis=0)
        scores = np.where(broadcast_mask, scores, -1e9)
    weights = stable_softmax(scores, axis=-1)
    return np.matmul(weights, value), weights


@dataclass(frozen=True)
class MiniTransformerConfig:
    vocab_size: int
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 128
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    max_length: int = 128
    seed: int = 13

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")


class Linear:
    def __init__(self, rng: np.random.Generator, in_features: int, out_features: int) -> None:
        scale = 1.0 / np.sqrt(in_features)
        self.weight = rng.normal(0.0, scale, size=(in_features, out_features))
        self.bias = np.zeros(out_features)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values @ self.weight + self.bias


class LayerNorm:
    def __init__(self, features: int, eps: float = 1e-5) -> None:
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.eps = eps

    def __call__(self, values: np.ndarray) -> np.ndarray:
        mean = values.mean(axis=-1, keepdims=True)
        variance = values.var(axis=-1, keepdims=True)
        return self.gamma * (values - mean) / np.sqrt(variance + self.eps) + self.beta


class FeedForward:
    def __init__(self, rng: np.random.Generator, d_model: int, d_ff: int) -> None:
        self.in_proj = Linear(rng, d_model, d_ff)
        self.out_proj = Linear(rng, d_ff, d_model)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        hidden = np.maximum(self.in_proj(values), 0.0)
        return self.out_proj(hidden)


class MultiHeadAttention:
    def __init__(self, rng: np.random.Generator, d_model: int, num_heads: int) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = Linear(rng, d_model, d_model)
        self.key = Linear(rng, d_model, d_model)
        self.value = Linear(rng, d_model, d_model)
        self.output = Linear(rng, d_model, d_model)

    def _split_heads(self, values: np.ndarray) -> np.ndarray:
        batch, length, _ = values.shape
        values = values.reshape(batch, length, self.num_heads, self.head_dim)
        return np.transpose(values, (0, 2, 1, 3))

    def _merge_heads(self, values: np.ndarray) -> np.ndarray:
        batch, heads, length, head_dim = values.shape
        values = np.transpose(values, (0, 2, 1, 3))
        return values.reshape(batch, length, heads * head_dim)

    def __call__(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        q = self._split_heads(self.query(query))
        k = self._split_heads(self.key(key))
        v = self._split_heads(self.value(value))
        attended, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return self.output(self._merge_heads(attended))


class EncoderLayer:
    def __init__(self, rng: np.random.Generator, config: MiniTransformerConfig) -> None:
        self.attention = MultiHeadAttention(rng, config.d_model, config.num_heads)
        self.feed_forward = FeedForward(rng, config.d_model, config.d_ff)
        self.norm_attention = LayerNorm(config.d_model)
        self.norm_ff = LayerNorm(config.d_model)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = self.norm_attention(values + self.attention(values, values, values))
        return self.norm_ff(values + self.feed_forward(values))


class DecoderLayer:
    def __init__(self, rng: np.random.Generator, config: MiniTransformerConfig) -> None:
        self.self_attention = MultiHeadAttention(rng, config.d_model, config.num_heads)
        self.cross_attention = MultiHeadAttention(rng, config.d_model, config.num_heads)
        self.feed_forward = FeedForward(rng, config.d_model, config.d_ff)
        self.norm_self = LayerNorm(config.d_model)
        self.norm_cross = LayerNorm(config.d_model)
        self.norm_ff = LayerNorm(config.d_model)

    def __call__(self, target: np.ndarray, encoded_source: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        target = self.norm_self(target + self.self_attention(target, target, target, mask=target_mask))
        target = self.norm_cross(target + self.cross_attention(target, encoded_source, encoded_source))
        return self.norm_ff(target + self.feed_forward(target))


class MiniSeq2SeqTransformer:
    """Small NumPy encoder-decoder Transformer for architecture inspection.

    This model is not intended to compete with pretrained summarizers. It exists
    to make attention, masking, positional encoding, and seq2seq logits visible.
    """

    def __init__(self, config: MiniTransformerConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.d_model)
        self.embedding = self.rng.normal(0.0, scale, size=(config.vocab_size, config.d_model))
        self.output_projection = Linear(self.rng, config.d_model, config.vocab_size)
        self.encoder_layers = [EncoderLayer(self.rng, config) for _ in range(config.num_encoder_layers)]
        self.decoder_layers = [DecoderLayer(self.rng, config) for _ in range(config.num_decoder_layers)]
        self.positions = positional_encoding(config.max_length, config.d_model)

    def _embed(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, length]")
        if token_ids.shape[1] > self.config.max_length:
            raise ValueError("sequence length exceeds configured max_length")
        clipped = np.clip(token_ids, 0, self.config.vocab_size - 1)
        return self.embedding[clipped] + self.positions[: token_ids.shape[1]]

    def encode(self, source_ids: np.ndarray) -> np.ndarray:
        encoded = self._embed(source_ids)
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        return encoded

    def forward(self, source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
        encoded = self.encode(source_ids)
        decoded = self._embed(target_ids)
        mask = causal_mask(target_ids.shape[1])
        for layer in self.decoder_layers:
            decoded = layer(decoded, encoded, mask)
        return self.output_projection(decoded)

    def greedy_decode(self, source_ids: np.ndarray, bos_id: int, eos_id: int, max_new_tokens: int) -> list[int]:
        generated = [bos_id]
        for _ in range(max_new_tokens):
            target = np.array([generated], dtype=np.int64)
            logits = self.forward(source_ids, target)
            next_id = int(np.argmax(logits[0, -1]))
            generated.append(next_id)
            if next_id == eos_id:
                break
        return generated

