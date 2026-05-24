"""Triton kernel: varlen RMSNorm (LLaMA / Mistral / Mixtral norm).

RMSNorm is per-token elementwise: ``y = x / sqrt(mean(x²) + eps) * weight``.
No cross-token interaction, so the kernel maps one program per token
and reduces only over the feature dim.

Compatible with the packed scree.Array layout — values are
``(total_tokens, feature_dim)``; the kernel doesn't even need offsets.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _varlen_rmsnorm_kernel(
    X,
    Y,
    W,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    eps,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per token. Reduce over the feature dim."""
    m = tl.program_id(0)
    d_idx = tl.arange(0, BLOCK_D)

    x_ptrs = X + m * stride_xm + d_idx * stride_xd
    x = tl.load(x_ptrs).to(tl.float32)

    # RMS = sqrt(mean(x²) + eps)
    rms = tl.sqrt(tl.sum(x * x) / BLOCK_D + eps)
    y = x / rms

    if HAS_WEIGHT:
        w = tl.load(W + d_idx).to(tl.float32)
        y = y * w

    y_ptrs = Y + m * stride_ym + d_idx * stride_yd
    tl.store(y_ptrs, y.to(Y.dtype.element_ty))


def varlen_rmsnorm_triton(x, weight=None, eps: float = 1e-6):
    """RMSNorm over the last dim, Triton implementation.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(total_tokens, feature_dim)`` or ``(total_tokens, *other, feature_dim)``.
        Will be reshaped to 2-D for the kernel. dtype fp16 / bf16 / fp32.
        Must live on a CUDA device.
    weight : torch.Tensor, optional
        Per-feature scale of shape ``(feature_dim,)``.
    eps : float
        Numerical-stability epsilon (typical 1e-6 for LLaMA-family).

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``x``.
    """
    import torch

    assert x.is_cuda, "varlen_rmsnorm_triton requires CUDA tensors"
    original_shape = x.shape
    feature_dim = original_shape[-1]
    x_flat = x.reshape(-1, feature_dim).contiguous()
    n_tokens = x_flat.shape[0]

    if weight is not None:
        assert weight.shape == (feature_dim,)
        assert weight.device == x.device
        weight = weight.to(x.dtype).contiguous()
        has_weight = True
    else:
        has_weight = False
        weight = x_flat  # dummy passthrough to satisfy signature

    y_flat = torch.empty_like(x_flat)
    grid = (n_tokens,)
    _varlen_rmsnorm_kernel[grid](
        x_flat,
        y_flat,
        weight,
        x_flat.stride(0),
        x_flat.stride(1),
        y_flat.stride(0),
        y_flat.stride(1),
        eps,
        HAS_WEIGHT=has_weight,
        BLOCK_D=feature_dim,
    )
    return y_flat.reshape(original_shape)
