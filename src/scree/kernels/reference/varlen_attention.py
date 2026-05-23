"""Reference (slow but correct) implementation of varlen self-attention."""

from __future__ import annotations

import math

from ..._core import Array, _is_torch


def varlen_attention(q: Array, k: Array, v: Array, causal: bool = False) -> Array:
    """Variable-length self-attention.

    Each sequence in the batch attends only to itself — no cross-sequence
    attention. This is the operation that powers FlashAttention-varlen and
    the packed inference path of vLLM/SGLang; here we ship the obviously
    correct slow reference for use as a ground truth in CI.

    Parameters
    ----------
    q, k, v : scree.Array
        Each with shape ``(total_len, n_heads, head_dim)`` and matching
        ``offsets``.
    causal : bool
        If True, apply a lower-triangular mask within each sequence.

    Returns
    -------
    scree.Array
        Same offsets as ``q``.
    """
    head_dim = q.values.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    if _is_torch(q.values):
        import torch

        if not torch.equal(q.offsets, k.offsets) or not torch.equal(q.offsets, v.offsets):
            raise ValueError("q, k, v must have identical offsets")

        out_rows = []
        for i in range(q.batch_size):
            s = int(q.offsets[i])
            e = int(q.offsets[i + 1])
            qi = q.values[s:e]  # (Li, H, D)
            ki = k.values[s:e]
            vi = v.values[s:e]
            scores = torch.einsum("ihd,jhd->hij", qi, ki) * scale  # (H, Li, Li)
            if causal:
                length = qi.shape[0]
                mask = torch.triu(
                    torch.ones(length, length, device=qi.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out_i = torch.einsum("hij,jhd->ihd", attn, vi)
            out_rows.append(out_i)
        values = torch.cat(out_rows, dim=0)
    else:
        import numpy as np

        if not np.array_equal(q.offsets, k.offsets) or not np.array_equal(q.offsets, v.offsets):
            raise ValueError("q, k, v must have identical offsets")

        out_rows = []
        for i in range(q.batch_size):
            s = int(q.offsets[i])
            e = int(q.offsets[i + 1])
            qi = q.values[s:e]
            ki = k.values[s:e]
            vi = v.values[s:e]
            scores = np.einsum("ihd,jhd->hij", qi, ki) * scale
            if causal:
                length = qi.shape[0]
                mask = np.triu(np.ones((length, length), dtype=bool), k=1)
                scores = np.where(mask, -np.inf, scores)
            scores_max = scores.max(axis=-1, keepdims=True)
            scores_exp = np.exp(scores - scores_max)
            attn = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
            out_i = np.einsum("hij,jhd->ihd", attn, vi)
            out_rows.append(out_i)
        values = np.concatenate(out_rows, axis=0)

    return Array(values=values, offsets=q.offsets, ragged_dim=0)
