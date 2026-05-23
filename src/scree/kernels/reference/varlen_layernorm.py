"""Reference (slow but correct) implementation of varlen layernorm.

Layernorm is per-token, so for variable-length data it's just elementwise
normalization over the last (feature) dim. No cross-row interaction —
the only reason this needs a varlen implementation is to operate directly
on a packed ``scree.Array`` without unpacking.
"""

from __future__ import annotations

from ..._core import Array, _is_torch


def varlen_layernorm(
    arr: Array,
    weight: object | None = None,
    bias: object | None = None,
    eps: float = 1e-5,
) -> Array:
    """LayerNorm over the last dim of a packed scree.Array.

    Parameters
    ----------
    arr : scree.Array
        Packed values of shape ``(total_len, ..., feature_dim)``.
    weight, bias : optional
        Scale and shift parameters of shape ``(feature_dim,)``.
    eps : float
        Numerical stability epsilon.
    """
    if _is_torch(arr.values):
        import torch

        x = arr.values
        feature_dim = x.shape[-1]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + eps)
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias
    else:
        import numpy as np

        x = arr.values
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + eps)
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias

    return Array(values=y, offsets=arr.offsets, ragged_dim=arr.ragged_dim)
