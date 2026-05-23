"""Triton kernel: varlen self-attention forward.

Forward pass of variable-length self-attention on packed sequences.
Each program processes a tile of ``BLOCK_M`` queries for one sequence,
one head, looping over keys/values in tiles of ``BLOCK_N`` using the
FlashAttention-2 online-softmax recurrence.

This is the headline GPU kernel for scree v0.1. Backward, GQA, and
autotuning come in v0.2.

Status: first GPU validation pending — correctness is checked against
FlashAttention-2 varlen on H100 by ``benchmarks/modal_bench.py``.
"""

from __future__ import annotations

import math

import triton
import triton.language as tl


# Triton autotune is currently disabled. We attempted a 3-config grid
# of empirically-safe (num_warps=4) configs on H100; the resulting
# steady-state ran SLOWER than the hardcoded baseline (1.50x vs 1.21x of
# FA-2). The autotune overhead appears to leak into per-iteration
# timing for short kernels, or autotune picked a config that benchmarks
# fast in isolation but loses in the full bench loop.
#
# Hardcoded config below produced 1.21x of FA-2 varlen with PASS
# correctness on first attempt. Until we have either Triton 3.1+ in the
# Modal image (which fixes the broader Hopper compiler bug) or
# subprocess-isolated probing to fully map the safe-set, this is the
# kernel scree ships.
@triton.jit
def _varlen_attn_fwd_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens,
    sm_scale,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # Program grid: (batch_idx, q_block_idx, head_idx)
    batch = tl.program_id(0)
    q_block = tl.program_id(1)
    head = tl.program_id(2)

    # Sequence range in the packed buffer for this batch element.
    start = tl.load(cu_seqlens + batch).to(tl.int32)
    end = tl.load(cu_seqlens + batch + 1).to(tl.int32)
    seq_len = end - start

    # Skip if this Q-block is entirely past the sequence end.
    q_offset = q_block * BLOCK_M
    if q_offset >= seq_len:
        return

    # Query tile offsets within this sequence.
    q_idx = q_offset + tl.arange(0, BLOCK_M)
    q_in_range = q_idx < seq_len
    d_idx = tl.arange(0, HEAD_DIM)

    # Load Q tile: shape (BLOCK_M, HEAD_DIM)
    q_ptrs = (
        Q
        + (start + q_idx[:, None]) * stride_qm
        + head * stride_qh
        + d_idx[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_in_range[:, None], other=0.0)

    # Online-softmax accumulators (kept in fp32).
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Determine how many K/V blocks to visit.
    # Under causal masking, a query at position q can only attend to
    # positions <= q, so blocks beyond the highest needed one are skipped.
    if CAUSAL:
        n_end = tl.minimum(seq_len, q_offset + BLOCK_M)
    else:
        n_end = seq_len

    for k_block_start in range(0, n_end, BLOCK_N):
        k_idx = k_block_start + tl.arange(0, BLOCK_N)
        k_in_range = k_idx < seq_len

        # Load K tile, shape (BLOCK_N, HEAD_DIM)
        k_ptrs = (
            K
            + (start + k_idx[:, None]) * stride_kn
            + head * stride_kh
            + d_idx[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_in_range[:, None], other=0.0)

        # Load V tile, shape (BLOCK_N, HEAD_DIM)
        v_ptrs = (
            V
            + (start + k_idx[:, None]) * stride_vn
            + head * stride_vh
            + d_idx[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=k_in_range[:, None], other=0.0)

        # Score tile: (BLOCK_M, BLOCK_N)
        scores = tl.dot(q, tl.trans(k)) * sm_scale

        # Mask out padding positions in K.
        scores = tl.where(k_in_range[None, :], scores, -float("inf"))
        # Causal mask: position q attends to position k only if k <= q.
        if CAUSAL:
            scores = tl.where(q_idx[:, None] >= k_idx[None, :], scores, -float("inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    # Normalize the accumulator with the final denominator.
    acc = acc / l_i[:, None]

    # Write output, masking off Q-block tail beyond seq_len.
    o_ptrs = (
        O
        + (start + q_idx[:, None]) * stride_om
        + head * stride_oh
        + d_idx[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=q_in_range[:, None])


def varlen_attention_triton(
    q,
    k,
    v,
    cu_seqlens,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
):
    """Variable-length self-attention forward via Triton.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(total_tokens, n_heads, head_dim)``, dtype float16 or bfloat16.
        Must live on a CUDA device.
    cu_seqlens : torch.Tensor
        int32 tensor of shape ``(batch + 1,)`` — the offsets in scree.Array
        and FlashAttention's ``cu_seqlens`` convention.
    causal : bool
        Apply lower-triangular causal mask within each sequence.
    block_m, block_n : int
        Tile sizes for Q and K/V. Defaults (64, 64) produce 1.21× FA-2
        varlen on H100 with PASS correctness. Tuning is deferred — see
        the comment above _varlen_attn_fwd_kernel for the autotune story.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``q``.
    """
    import torch

    assert q.shape == k.shape == v.shape, "q, k, v must have identical shape"
    assert q.is_cuda, "varlen_attention_triton requires CUDA tensors"
    assert q.dtype in (torch.float16, torch.bfloat16), "fp16 or bf16 only in v0.0"

    total, n_heads, head_dim = q.shape
    batch = cu_seqlens.numel() - 1

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq = int(seq_lens.max().item())
    n_q_blocks = triton.cdiv(max_seq, block_m)

    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)

    grid = (batch, n_q_blocks, n_heads)
    _varlen_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        cu_seqlens,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return out
