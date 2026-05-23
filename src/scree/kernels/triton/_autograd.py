"""torch.autograd.Function wrapper around varlen_attention_triton.

Forward and backward both use Triton kernels:
- Forward: ``varlen_attention_triton`` (FA-2 style online softmax)
- Backward: ``varlen_attention_triton_backward`` (FA-2 style: preprocess + dKV + dQ)

Usage
-----
    from scree.kernels.triton import varlen_attention_triton_autograd
    out = varlen_attention_triton_autograd(q, k, v, cu_seqlens, causal=True)
    out.sum().backward()   # gradients flow into q, k, v
"""

from __future__ import annotations


def _build_autograd_function():
    """Build the autograd.Function lazily so the module imports without torch."""
    import torch

    from ._backward import varlen_attention_triton_backward
    from .varlen_attention import varlen_attention_triton

    class _VarlenAttentionTriton(torch.autograd.Function):
        """Full Triton forward + backward — FA-2 style throughout."""

        @staticmethod
        def forward(ctx, q, k, v, cu_seqlens, causal):
            out, lse = varlen_attention_triton(
                q, k, v, cu_seqlens, causal=causal, return_lse=True
            )
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
            ctx.causal = causal
            return out

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
            causal = ctx.causal
            # grad_output might not be contiguous; the kernel expects it to be.
            grad_output = grad_output.contiguous()
            dq, dk, dv = varlen_attention_triton_backward(
                grad_output, q, k, v, out, lse, cu_seqlens, causal=causal
            )
            # No gradient for cu_seqlens or causal.
            return dq, dk, dv, None, None

    return _VarlenAttentionTriton


def varlen_attention_triton_autograd(q, k, v, cu_seqlens, causal: bool = False):
    """Autograd-aware wrapper around the Triton varlen attention.

    Both forward and backward run on dedicated Triton kernels (FA-2 style:
    online softmax in forward, preprocess + dKV + dQ in backward).

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(total_tokens, n_heads, head_dim)``, fp16/bf16, CUDA.
    cu_seqlens : torch.Tensor
        int32 ``(batch + 1,)``.
    causal : bool
        Lower-triangular causal mask within each sequence.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``q``, autograd-compatible.
    """
    fn = _build_autograd_function()
    return fn.apply(q, k, v, cu_seqlens, causal)
