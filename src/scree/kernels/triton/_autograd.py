"""torch.autograd.Function wrapper around varlen_attention_triton.

Lets users get the Triton forward speed AND working backpropagation in
one call. The backward pass currently routes through the reference
(slow but correct) implementation; the FA-2 style Triton backward
lands in v0.1.

Usage
-----
    from scree.kernels.triton import varlen_attention_triton_autograd
    out = varlen_attention_triton_autograd(q, k, v, cu_seqlens, causal=True)
    out.sum().backward()   # gradients flow into q, k, v

This wrapper exists because the raw `varlen_attention_triton` is a
plain function call, not autograd-aware — calling `.backward()` on its
output raises. With this wrapper, training loops Just Work.
"""

from __future__ import annotations

import math
from typing import Any


def _build_autograd_function():
    """Build the autograd.Function lazily so the module imports without torch."""
    import torch

    from .varlen_attention import varlen_attention_triton

    class _VarlenAttentionTriton(torch.autograd.Function):
        """Forward: Triton kernel (fast). Backward: reference impl (correct, slow)."""

        @staticmethod
        def forward(ctx, q, k, v, cu_seqlens, causal):
            ctx.save_for_backward(q, k, v, cu_seqlens)
            ctx.causal = causal
            return varlen_attention_triton(q, k, v, cu_seqlens, causal=causal)

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v, cu_seqlens = ctx.saved_tensors
            causal = ctx.causal

            # Recompute forward through the reference path with autograd enabled,
            # then call .backward() to populate gradients.
            q_ref = q.detach().clone().requires_grad_(True)
            k_ref = k.detach().clone().requires_grad_(True)
            v_ref = v.detach().clone().requires_grad_(True)

            # Inline the reference math for varlen attention so we don't pull
            # in scree._core (which would create a circular import dependency).
            head_dim = q_ref.shape[-1]
            scale = 1.0 / math.sqrt(head_dim)
            batch = cu_seqlens.numel() - 1

            out_rows = []
            for i in range(batch):
                s = int(cu_seqlens[i].item())
                e = int(cu_seqlens[i + 1].item())
                qi = q_ref[s:e]
                ki = k_ref[s:e]
                vi = v_ref[s:e]
                scores = torch.einsum("ihd,jhd->hij", qi, ki) * scale
                if causal:
                    length = qi.shape[0]
                    mask = torch.triu(
                        torch.ones(length, length, device=qi.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    scores = scores.masked_fill(mask, float("-inf"))
                attn = torch.softmax(scores.float(), dim=-1).to(qi.dtype)
                out_rows.append(torch.einsum("hij,jhd->ihd", attn, vi))
            out = torch.cat(out_rows, dim=0)

            out.backward(grad_output)
            # Cu_seqlens and causal don't take gradients.
            return q_ref.grad, k_ref.grad, v_ref.grad, None, None

    return _VarlenAttentionTriton


def varlen_attention_triton_autograd(q, k, v, cu_seqlens, causal: bool = False):
    """Autograd-aware wrapper around ``varlen_attention_triton``.

    Forward uses the Triton kernel (fast). Backward uses the reference
    PyTorch implementation (correct, slower) — full Triton backward lands
    in v0.1.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(total_tokens, n_heads, head_dim)``, fp16/bf16, CUDA.
    cu_seqlens : torch.Tensor
        int32 ``(batch + 1,)`` — same convention as the raw kernel.
    causal : bool
        Lower-triangular causal mask within each sequence.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``q``, with gradients flowing on backward.
    """
    fn = _build_autograd_function()
    return fn.apply(q, k, v, cu_seqlens, causal)
