"""A transformer block that never pads.

The full pre-norm transformer forward pass — varlen self-attention,
varlen layernorm, feedforward, residuals — operating on scree.Arrays
throughout. There is no ``attention_mask`` to thread through and no
FLOPs spent on padding positions.

Run with:
    python examples/02_no_pad_transformer.py
"""

from __future__ import annotations

import numpy as np

import scree
from scree.kernels.reference import varlen_attention, varlen_layernorm


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def feedforward(arr: scree.Array, W1: np.ndarray, W2: np.ndarray) -> scree.Array:
    """Two-layer MLP with GELU, operating directly on the packed buffer."""
    h = gelu(arr.values @ W1)
    out = h @ W2
    return scree.Array(values=out, offsets=arr.offsets, ragged_dim=0)


def transformer_block(
    x: scree.Array,
    Wqkv: np.ndarray,
    Wo: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    n_heads: int,
) -> scree.Array:
    """One pre-norm transformer block: norm → varlen attn → residual → norm → ff → residual."""
    model_dim = x.values.shape[-1]
    head_dim = model_dim // n_heads

    # Attention sub-block
    h = varlen_layernorm(x)
    qkv = (h.values @ Wqkv).reshape(-1, 3, n_heads, head_dim)
    q = scree.Array(values=qkv[:, 0], offsets=x.offsets)
    k = scree.Array(values=qkv[:, 1], offsets=x.offsets)
    v = scree.Array(values=qkv[:, 2], offsets=x.offsets)
    attn_out = varlen_attention(q, k, v, causal=True)
    attn_proj = attn_out.values.reshape(-1, model_dim) @ Wo
    h1 = scree.Array(values=x.values + attn_proj, offsets=x.offsets)

    # Feedforward sub-block
    ff_out = feedforward(varlen_layernorm(h1), W1, W2)
    return scree.Array(values=h1.values + ff_out.values, offsets=x.offsets)


def main() -> None:
    rng = np.random.default_rng(0)
    lengths = [40, 12, 50, 25]
    model_dim, n_heads, ff_dim = 128, 4, 256

    seqs = [rng.standard_normal((n, model_dim)).astype(np.float32) for n in lengths]
    x = scree.pack(seqs)

    scale = 1.0 / np.sqrt(model_dim)
    Wqkv = (rng.standard_normal((model_dim, 3 * model_dim)) * scale).astype(np.float32)
    Wo = (rng.standard_normal((model_dim, model_dim)) * scale).astype(np.float32)
    W1 = (rng.standard_normal((model_dim, ff_dim)) * scale).astype(np.float32)
    W2 = (rng.standard_normal((ff_dim, model_dim)) * (1.0 / np.sqrt(ff_dim))).astype(np.float32)

    print(f"input:  {x}")
    out = transformer_block(x, Wqkv, Wo, W1, W2, n_heads=n_heads)
    print(f"output: {out}")

    assert out.values.shape == x.values.shape
    assert (out.offsets == x.offsets).all()

    for i, row in enumerate(scree.unpack(out)):
        print(f"  seq {i}: shape={row.shape}")

    padded_tokens = len(lengths) * max(lengths)
    print(f"\ntotal tokens processed: {x.total_length}")
    print(f"if padded:              {padded_tokens}")
    print(f"compute saved:          {(1 - x.total_length / padded_tokens) * 100:.0f}%")


if __name__ == "__main__":
    main()
