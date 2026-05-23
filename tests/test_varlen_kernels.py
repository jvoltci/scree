"""Tests for the reference varlen kernels.

The reference impls are the ground truth that the future Triton kernels
must match bit-exactly (within FP tolerance).
"""

import numpy as np
import pytest

import scree
from scree.kernels.reference import varlen_attention, varlen_layernorm, varlen_softmax


def test_varlen_attention_matches_per_sequence_baseline_numpy():
    """Varlen attention output for each row equals doing attention on
    that row in isolation."""
    rng = np.random.default_rng(0)
    n_heads, head_dim = 2, 4
    lengths = [3, 5, 2]
    seqs_q = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_k = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_v = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]

    q = scree.pack(seqs_q)
    k = scree.pack(seqs_k)
    v = scree.pack(seqs_v)
    out = varlen_attention(q, k, v, causal=False)
    out_rows = scree.unpack(out)

    for i in range(len(lengths)):
        qi, ki, vi = seqs_q[i], seqs_k[i], seqs_v[i]
        scale = 1.0 / np.sqrt(head_dim)
        scores = np.einsum("ihd,jhd->hij", qi, ki) * scale
        scores_max = scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores - scores_max)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        baseline = np.einsum("hij,jhd->ihd", attn, vi)
        np.testing.assert_allclose(out_rows[i], baseline, atol=1e-5)


def test_varlen_attention_causal_first_token_is_v0():
    """Under causal mask, the first token can only attend to itself,
    so its output equals its v value."""
    rng = np.random.default_rng(0)
    lengths = [4, 6]
    n_heads, head_dim = 1, 4
    seqs_q = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_k = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_v = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]

    q = scree.pack(seqs_q)
    k = scree.pack(seqs_k)
    v = scree.pack(seqs_v)
    out = varlen_attention(q, k, v, causal=True)
    out_rows = scree.unpack(out)

    for i in range(len(lengths)):
        np.testing.assert_allclose(out_rows[i][0], seqs_v[i][0], atol=1e-5)


def test_varlen_attention_rejects_mismatched_offsets():
    rng = np.random.default_rng(0)
    q_seqs = [rng.standard_normal((n, 1, 4)).astype(np.float32) for n in [3, 5]]
    k_seqs = [rng.standard_normal((n, 1, 4)).astype(np.float32) for n in [4, 4]]  # different
    v_seqs = k_seqs
    q = scree.pack(q_seqs)
    k = scree.pack(k_seqs)
    v = scree.pack(v_seqs)
    with pytest.raises(ValueError, match=r"identical offsets"):
        varlen_attention(q, k, v)


def test_varlen_layernorm_matches_padded_baseline():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 8)).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    out = varlen_layernorm(arr, eps=1e-5)
    out_rows = scree.unpack(out)

    for i, seq in enumerate(seqs):
        mean = seq.mean(axis=-1, keepdims=True)
        var = seq.var(axis=-1, keepdims=True)
        baseline = (seq - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(out_rows[i], baseline, atol=1e-5)


def test_varlen_softmax_per_sequence_sums_to_one():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n,)).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    out = varlen_softmax(arr)
    out_rows = scree.unpack(out)
    for row in out_rows:
        np.testing.assert_allclose(row.sum(), 1.0, atol=1e-5)


def test_varlen_softmax_matches_per_sequence_baseline():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 3)).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    out = varlen_softmax(arr)
    out_rows = scree.unpack(out)
    for i, seq in enumerate(seqs):
        # Softmax along axis 0 (the ragged dim)
        row_max = seq.max(axis=0, keepdims=True)
        row_exp = np.exp(seq - row_max)
        baseline = row_exp / row_exp.sum(axis=0, keepdims=True)
        np.testing.assert_allclose(out_rows[i], baseline, atol=1e-5)
