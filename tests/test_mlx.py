"""MLX backend tests — exercises the Apple Silicon path end-to-end.

MLX runs natively on M-series GPUs via Metal. Tests are skipped on
platforms where MLX is unavailable.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")  # type: ignore

import scree
from scree.kernels.reference import varlen_attention, varlen_layernorm, varlen_softmax


def _mlx_close(actual, expected, atol=1e-5):
    """Compare an MLX array to a numpy array."""
    a = np.array(actual)
    e = np.array(expected)
    np.testing.assert_allclose(a, e, atol=atol)


def test_pack_unpack_roundtrip_mlx():
    seqs = [mx.random.normal((n, 4)) for n in [3, 5, 2, 7]]
    arr = scree.pack(seqs)
    assert arr.batch_size == 4
    assert arr.total_length == 17
    out = scree.unpack(arr)
    for a, b in zip(seqs, out):
        _mlx_close(b, a)


def test_to_from_padded_roundtrip_mlx():
    seqs = [mx.random.normal((n, 4)) for n in [3, 5, 2, 7]]
    arr = scree.pack(seqs)
    padded, mask = scree.to_padded(arr)
    assert padded.shape == (4, 7, 4)
    assert mask.shape == (4, 7)
    arr2 = scree.from_padded(padded, mask)
    _mlx_close(arr2.values, arr.values)
    _mlx_close(arr2.offsets, arr.offsets)


def test_varlen_attention_mlx_matches_numpy():
    """MLX varlen_attention output matches the NumPy reference numerically.

    Tolerance is loose because MLX matmul on Apple Silicon uses tensor-core
    mixed precision internally even on float32 inputs (similar to NVIDIA
    TF32). A single matmul accumulates ~1e-3 deviation; chained ops in
    attention can compound to ~5e-3.
    """
    rng = np.random.default_rng(0)
    n_heads, head_dim = 2, 4
    lengths = [3, 5, 2]
    seqs_q = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_k = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_v = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]

    # NumPy baseline
    q_np = scree.pack(seqs_q)
    k_np = scree.pack(seqs_k)
    v_np = scree.pack(seqs_v)
    out_np = varlen_attention(q_np, k_np, v_np, causal=True)

    # MLX path
    q_mx = scree.pack([mx.array(s) for s in seqs_q])
    k_mx = scree.pack([mx.array(s) for s in seqs_k])
    v_mx = scree.pack([mx.array(s) for s in seqs_v])
    out_mx = varlen_attention(q_mx, k_mx, v_mx, causal=True)

    _mlx_close(out_mx.values, out_np.values, atol=5e-3)


def test_varlen_layernorm_mlx_matches_numpy():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 8)).astype(np.float32) for n in [3, 5, 2]]

    out_np = varlen_layernorm(scree.pack(seqs), eps=1e-5)
    out_mx = varlen_layernorm(scree.pack([mx.array(s) for s in seqs]), eps=1e-5)
    _mlx_close(out_mx.values, out_np.values, atol=1e-4)


def test_varlen_softmax_mlx_matches_numpy():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 3)).astype(np.float32) for n in [3, 5, 2]]

    out_np = varlen_softmax(scree.pack(seqs))
    out_mx = varlen_softmax(scree.pack([mx.array(s) for s in seqs]))
    _mlx_close(out_mx.values, out_np.values, atol=1e-5)
