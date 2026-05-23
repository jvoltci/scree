"""Reference (slow but correct) implementation of varlen RMSNorm.

RMSNorm (Zhang & Sennrich, 2019) drops the mean-subtraction step from
LayerNorm — it normalizes by the root-mean-square only. It is the norm
used by LLaMA, Mistral, Mixtral, DeepSeek, Qwen, and most modern open
transformers, replacing LayerNorm in nearly every architecture released
since 2023.

Like LayerNorm, RMSNorm is per-token (no cross-row interaction), so for
variable-length data it's just elementwise on the packed buffer.
"""

from __future__ import annotations

from ..._core import Array, _is_mlx, _is_torch


def varlen_rmsnorm(
    arr: Array,
    weight: object | None = None,
    eps: float = 1e-6,
) -> Array:
    """RMSNorm over the last dim of a packed scree.Array.

    Parameters
    ----------
    arr : scree.Array
        Packed values of shape ``(total_len, ..., feature_dim)``.
    weight : optional
        Scale parameter of shape ``(feature_dim,)``.
    eps : float
        Numerical stability epsilon (typical: 1e-6 for LLaMA-family).
    """
    if _is_mlx(arr.values):
        import mlx.core as mx

        x = arr.values
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
        y = x / rms
        if weight is not None:
            y = y * weight
    elif _is_torch(arr.values):
        import torch

        x = arr.values
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = x / rms
        if weight is not None:
            y = y * weight
    else:
        import numpy as np

        x = arr.values
        rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
        y = x / rms
        if weight is not None:
            y = y * weight

    return Array(values=y, offsets=arr.offsets, ragged_dim=arr.ragged_dim)
