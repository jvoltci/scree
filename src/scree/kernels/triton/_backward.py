"""Triton backward kernels for varlen self-attention.

Implements the FlashAttention-2 style backward pass using three kernels:

  1. ``_varlen_attn_bwd_preprocess_kernel`` — computes Delta[i, h] = sum_d O[i, h, d] * dO[i, h, d]
     per-token, per-head. This is the row-wise dot-product needed by the
     softmax backward identity (avoids recomputing attention probs twice).

  2. ``_varlen_attn_bwd_dkv_kernel`` — for each K/V tile, iterates over
     valid Q tiles and accumulates dK and dV by recomputing the attention
     probabilities P from saved LSE.

  3. ``_varlen_attn_bwd_dq_kernel`` — for each Q tile, iterates over valid
     K/V tiles and accumulates dQ the same way.

The split into two backward kernels (dKV and dQ) avoids the need for
atomic accumulation: each kernel writes to a disjoint slice of its
respective gradient tensor.

These kernels work in tandem with the modified forward kernel (which
saves LSE per Q-position) to provide a complete autograd-aware Triton
implementation of varlen attention.

Status: untested without GPU access. Validated end-to-end via
``benchmarks/modal_backward_bench.py``.
"""

from __future__ import annotations

import math

import triton
import triton.language as tl


@triton.jit
def _varlen_attn_bwd_preprocess_kernel(
    O,
    dO,
    Delta,
    n_heads,
    stride_om,
    stride_oh,
    stride_od,
    stride_dom,
    stride_doh,
    stride_dod,
    stride_dm,
    stride_dh,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Per-token reduction: Delta[m, h] = sum_d O[m, h, d] * dO[m, h, d]."""
    # Grid: (m_block_idx, head_idx)
    m_block = tl.program_id(0)
    head = tl.program_id(1)

    m_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    d_idx = tl.arange(0, HEAD_DIM)

    # No mask: caller sizes the grid to cover exactly total_tokens.
    o_ptrs = O + m_idx[:, None] * stride_om + head * stride_oh + d_idx[None, :] * stride_od
    do_ptrs = dO + m_idx[:, None] * stride_dom + head * stride_doh + d_idx[None, :] * stride_dod
    o = tl.load(o_ptrs)
    do = tl.load(do_ptrs)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    delta_ptrs = Delta + m_idx * stride_dm + head * stride_dh
    tl.store(delta_ptrs, delta)


@triton.jit
def _varlen_attn_bwd_dkv_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    LSE,
    Delta,
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
    stride_dom,
    stride_doh,
    stride_dod,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    stride_lse_m,
    stride_lse_h,
    stride_delta_m,
    stride_delta_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Compute dK, dV for one K/V tile by iterating over valid Q tiles."""
    # Grid: (batch, k_block_idx, head)
    batch = tl.program_id(0)
    k_block = tl.program_id(1)
    head = tl.program_id(2)

    start = tl.load(cu_seqlens + batch).to(tl.int32)
    end = tl.load(cu_seqlens + batch + 1).to(tl.int32)
    seq_len = end - start

    k_offset = k_block * BLOCK_N
    if k_offset >= seq_len:
        return

    k_idx = k_offset + tl.arange(0, BLOCK_N)
    k_in_range = k_idx < seq_len
    d_idx = tl.arange(0, HEAD_DIM)

    # Load K and V tiles
    k_ptrs = K + (start + k_idx[:, None]) * stride_kn + head * stride_kh + d_idx[None, :] * stride_kd
    v_ptrs = V + (start + k_idx[:, None]) * stride_vn + head * stride_vh + d_idx[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=k_in_range[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=k_in_range[:, None], other=0.0)

    # Accumulators (fp32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Under causal: a Q at position q only attends to K at positions <= q.
    # So K at position k contributes to dK/dV only for Q at positions >= k.
    if CAUSAL:
        m_start = k_offset
    else:
        m_start = 0

    for m_block_start in range(m_start, seq_len, BLOCK_M):
        q_idx = m_block_start + tl.arange(0, BLOCK_M)
        q_in_range = q_idx < seq_len

        q_ptrs = Q + (start + q_idx[:, None]) * stride_qm + head * stride_qh + d_idx[None, :] * stride_qd
        do_ptrs = dO + (start + q_idx[:, None]) * stride_dom + head * stride_doh + d_idx[None, :] * stride_dod
        q = tl.load(q_ptrs, mask=q_in_range[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=q_in_range[:, None], other=0.0)

        # Load LSE and Delta for this Q tile
        lse_ptrs = LSE + (start + q_idx) * stride_lse_m + head * stride_lse_h
        delta_ptrs = Delta + (start + q_idx) * stride_delta_m + head * stride_delta_h
        lse = tl.load(lse_ptrs, mask=q_in_range, other=0.0)
        delta = tl.load(delta_ptrs, mask=q_in_range, other=0.0)

        # Recompute attention probabilities P
        # scores = q @ k^T * sm_scale; P = exp(scores - lse_q)
        scores = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M, BLOCK_N)
        # Mask out padding K positions
        scores = tl.where(k_in_range[None, :], scores, -float("inf"))
        if CAUSAL:
            scores = tl.where(q_idx[:, None] >= k_idx[None, :], scores, -float("inf"))
        # Mask out padding Q positions (their lse is garbage)
        scores = tl.where(q_in_range[:, None], scores, -float("inf"))

        p = tl.exp(scores - lse[:, None])  # (BLOCK_M, BLOCK_N), fp32

        # dV += P^T @ dO
        dv += tl.dot(tl.trans(p.to(do.dtype)), do).to(tl.float32)

        # dP = dO @ V^T  (BLOCK_M, BLOCK_N)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        # dS = P * (dP - delta_q)
        ds = p * (dp - delta[:, None])
        # dK += dS^T @ Q * sm_scale
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32) * sm_scale

    # Write dK, dV
    dk_ptrs = dK + (start + k_idx[:, None]) * stride_dkn + head * stride_dkh + d_idx[None, :] * stride_dkd
    dv_ptrs = dV + (start + k_idx[:, None]) * stride_dvn + head * stride_dvh + d_idx[None, :] * stride_dvd
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=k_in_range[:, None])
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=k_in_range[:, None])


@triton.jit
def _varlen_attn_bwd_dq_kernel(
    Q,
    K,
    V,
    dO,
    dQ,
    LSE,
    Delta,
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
    stride_dom,
    stride_doh,
    stride_dod,
    stride_dqm,
    stride_dqh,
    stride_dqd,
    stride_lse_m,
    stride_lse_h,
    stride_delta_m,
    stride_delta_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Compute dQ for one Q tile by iterating over valid K/V tiles."""
    batch = tl.program_id(0)
    q_block = tl.program_id(1)
    head = tl.program_id(2)

    start = tl.load(cu_seqlens + batch).to(tl.int32)
    end = tl.load(cu_seqlens + batch + 1).to(tl.int32)
    seq_len = end - start

    q_offset = q_block * BLOCK_M
    if q_offset >= seq_len:
        return

    q_idx = q_offset + tl.arange(0, BLOCK_M)
    q_in_range = q_idx < seq_len
    d_idx = tl.arange(0, HEAD_DIM)

    # Load Q, dO, LSE, Delta for this tile
    q_ptrs = Q + (start + q_idx[:, None]) * stride_qm + head * stride_qh + d_idx[None, :] * stride_qd
    do_ptrs = dO + (start + q_idx[:, None]) * stride_dom + head * stride_doh + d_idx[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=q_in_range[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=q_in_range[:, None], other=0.0)

    lse_ptrs = LSE + (start + q_idx) * stride_lse_m + head * stride_lse_h
    delta_ptrs = Delta + (start + q_idx) * stride_delta_m + head * stride_delta_h
    lse = tl.load(lse_ptrs, mask=q_in_range, other=0.0)
    delta = tl.load(delta_ptrs, mask=q_in_range, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # K/V range: causal restricts to k <= max(q in tile).
    if CAUSAL:
        n_end = tl.minimum(seq_len, q_offset + BLOCK_M)
    else:
        n_end = seq_len

    for k_block_start in range(0, n_end, BLOCK_N):
        k_idx = k_block_start + tl.arange(0, BLOCK_N)
        k_in_range = k_idx < seq_len

        k_ptrs = K + (start + k_idx[:, None]) * stride_kn + head * stride_kh + d_idx[None, :] * stride_kd
        v_ptrs = V + (start + k_idx[:, None]) * stride_vn + head * stride_vh + d_idx[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=k_in_range[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_in_range[:, None], other=0.0)

        # Recompute P
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        scores = tl.where(k_in_range[None, :], scores, -float("inf"))
        if CAUSAL:
            scores = tl.where(q_idx[:, None] >= k_idx[None, :], scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])

        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        # dS = P * (dP - delta_q)
        ds = p * (dp - delta[:, None])
        # dQ += dS @ K * sm_scale
        dq += tl.dot(ds.to(k.dtype), k).to(tl.float32) * sm_scale

    dq_ptrs = dQ + (start + q_idx[:, None]) * stride_dqm + head * stride_dqh + d_idx[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=q_in_range[:, None])


def varlen_attention_triton_backward(
    do,
    q,
    k,
    v,
    o,
    lse,
    cu_seqlens,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
):
    """Compute dQ, dK, dV for varlen self-attention via three Triton kernels.

    Parameters
    ----------
    do : (total, n_heads, head_dim) — gradient of the loss w.r.t. forward output
    q, k, v, o : (total, n_heads, head_dim) — forward inputs and output
    lse : (total, n_heads) fp32 — log-sum-exp saved from the forward pass
    cu_seqlens : (batch+1,) int32 — same as in the forward
    causal : bool
    block_m, block_n : int

    Returns
    -------
    (dq, dk, dv) : three tensors, same shape/dtype as q
    """
    import torch

    total, n_heads, head_dim = q.shape
    batch = cu_seqlens.numel() - 1

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq = int(seq_lens.max().item())

    # Output buffers
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = torch.empty((total, n_heads), dtype=torch.float32, device=q.device)

    sm_scale = 1.0 / math.sqrt(head_dim)

    # 1. Preprocess: compute Delta = sum(O * dO, dim=-1) per (token, head)
    # The grid covers exactly `total` tokens; we use a fixed BLOCK_M and pad
    # to a multiple of BLOCK_M by allocating dummy storage if needed.
    PRE_BLOCK = 128
    # Pad O and dO to multiple of PRE_BLOCK to avoid out-of-bounds reads.
    pad_total = ((total + PRE_BLOCK - 1) // PRE_BLOCK) * PRE_BLOCK
    if pad_total > total:
        # Use full-size buffers padded with zeros for the preprocess.
        o_pad = torch.zeros((pad_total, n_heads, head_dim), dtype=o.dtype, device=o.device)
        do_pad = torch.zeros((pad_total, n_heads, head_dim), dtype=do.dtype, device=do.device)
        delta_pad = torch.empty((pad_total, n_heads), dtype=torch.float32, device=q.device)
        o_pad[:total].copy_(o)
        do_pad[:total].copy_(do)
        n_blocks_m = pad_total // PRE_BLOCK
        _varlen_attn_bwd_preprocess_kernel[(n_blocks_m, n_heads)](
            o_pad, do_pad, delta_pad, n_heads,
            o_pad.stride(0), o_pad.stride(1), o_pad.stride(2),
            do_pad.stride(0), do_pad.stride(1), do_pad.stride(2),
            delta_pad.stride(0), delta_pad.stride(1),
            BLOCK_M=PRE_BLOCK, HEAD_DIM=head_dim,
        )
        delta = delta_pad[:total].contiguous()
    else:
        n_blocks_m = total // PRE_BLOCK
        _varlen_attn_bwd_preprocess_kernel[(n_blocks_m, n_heads)](
            o, do, delta, n_heads,
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            delta.stride(0), delta.stride(1),
            BLOCK_M=PRE_BLOCK, HEAD_DIM=head_dim,
        )

    # 2. dKV kernel
    n_k_blocks = triton.cdiv(max_seq, block_n)
    grid_kv = (batch, n_k_blocks, n_heads)
    _varlen_attn_bwd_dkv_kernel[grid_kv](
        q, k, v, do, dk, dv, lse, delta, cu_seqlens, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=head_dim, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )

    # 3. dQ kernel
    n_q_blocks = triton.cdiv(max_seq, block_m)
    grid_q = (batch, n_q_blocks, n_heads)
    _varlen_attn_bwd_dq_kernel[grid_q](
        q, k, v, do, dq, lse, delta, cu_seqlens, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=head_dim, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )

    return dq, dk, dv
