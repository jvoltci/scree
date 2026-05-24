"""Triton kernel: varlen LayerNorm.

LayerNorm: ``y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias``.
Per-token elementwise — no cross-token interaction. Kernel maps one
program per token and reduces only over the feature dim.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _varlen_layernorm_kernel(
    X,
    Y,
    W,
    B,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per token."""
    m = tl.program_id(0)
    d_idx = tl.arange(0, BLOCK_D)

    x_ptrs = X + m * stride_xm + d_idx * stride_xd
    x = tl.load(x_ptrs).to(tl.float32)

    mean = tl.sum(x) / BLOCK_D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / BLOCK_D
    y = x_centered / tl.sqrt(var + eps)

    if HAS_WEIGHT:
        w = tl.load(W + d_idx).to(tl.float32)
        y = y * w
    if HAS_BIAS:
        b = tl.load(B + d_idx).to(tl.float32)
        y = y + b

    y_ptrs = Y + m * stride_ym + d_idx * stride_yd
    tl.store(y_ptrs, y.to(Y.dtype.element_ty))


def varlen_layernorm_triton(x, weight=None, bias=None, eps: float = 1e-5):
    """LayerNorm over the last dim, Triton implementation.

    Parameters
    ----------
    x : torch.Tensor
        ``(total_tokens, feature_dim)`` or higher-rank with feature_dim last.
        Must live on a CUDA device.
    weight : torch.Tensor, optional
        Per-feature scale ``(feature_dim,)``.
    bias : torch.Tensor, optional
        Per-feature bias ``(feature_dim,)``.
    eps : float
        Numerical-stability epsilon (typical 1e-5).

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``x``.
    """
    import torch

    assert x.is_cuda, "varlen_layernorm_triton requires CUDA tensors"
    original_shape = x.shape
    feature_dim = original_shape[-1]
    x_flat = x.reshape(-1, feature_dim).contiguous()
    n_tokens = x_flat.shape[0]

    has_weight = weight is not None
    has_bias = bias is not None
    if has_weight:
        assert weight.shape == (feature_dim,) and weight.device == x.device
        weight = weight.to(x.dtype).contiguous()
    else:
        weight = x_flat
    if has_bias:
        assert bias.shape == (feature_dim,) and bias.device == x.device
        bias = bias.to(x.dtype).contiguous()
    else:
        bias = x_flat

    y_flat = torch.empty_like(x_flat)
    grid = (n_tokens,)
    _varlen_layernorm_kernel[grid](
        x_flat,
        y_flat,
        weight,
        bias,
        x_flat.stride(0),
        x_flat.stride(1),
        y_flat.stride(0),
        y_flat.stride(1),
        eps,
        HAS_WEIGHT=has_weight,
        HAS_BIAS=has_bias,
        BLOCK_D=feature_dim,
    )
    return y_flat.reshape(original_shape)
