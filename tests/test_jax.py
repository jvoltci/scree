"""JAX backend tests — exercises the JAX path end-to-end.

JAX uses immutable arrays, so pack/to_padded/from_padded follow the same
mutation-free pattern as MLX. Cross-backend numerical comparison against
NumPy is included as the correctness anchor.
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")  # type: ignore

import scree
from scree.kernels.reference import (
    varlen_attention,
    varlen_layernorm,
    varlen_rmsnorm,
    varlen_softmax,
)


def _jax_close(actual, expected, atol=1e-5):
    a = np.array(actual)
    e = np.array(expected)
    np.testing.assert_allclose(a, e, atol=atol)


def test_pack_unpack_roundtrip_jax():
    rng = np.random.default_rng(0)
    seqs = [jnp.array(rng.standard_normal((n, 4)).astype(np.float32)) for n in [3, 5, 2, 7]]
    arr = scree.pack(seqs)
    assert arr.batch_size == 4
    assert arr.total_length == 17
    out = scree.unpack(arr)
    for a, b in zip(seqs, out):
        _jax_close(b, a)


def test_to_from_padded_roundtrip_jax():
    rng = np.random.default_rng(0)
    seqs = [jnp.array(rng.standard_normal((n, 4)).astype(np.float32)) for n in [3, 5, 2, 7]]
    arr = scree.pack(seqs)
    padded, mask = scree.to_padded(arr)
    assert padded.shape == (4, 7, 4)
    assert mask.shape == (4, 7)
    arr2 = scree.from_padded(padded, mask)
    _jax_close(arr2.values, arr.values)
    _jax_close(arr2.offsets, arr.offsets)


def test_varlen_attention_jax_matches_numpy():
    """JAX varlen_attention output matches the NumPy reference."""
    rng = np.random.default_rng(0)
    n_heads, head_dim = 2, 4
    lengths = [3, 5, 2]
    seqs_q = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_k = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_v = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]

    q_np = scree.pack(seqs_q)
    k_np = scree.pack(seqs_k)
    v_np = scree.pack(seqs_v)
    out_np = varlen_attention(q_np, k_np, v_np, causal=True)

    q_jax = scree.pack([jnp.array(s) for s in seqs_q])
    k_jax = scree.pack([jnp.array(s) for s in seqs_k])
    v_jax = scree.pack([jnp.array(s) for s in seqs_v])
    out_jax = varlen_attention(q_jax, k_jax, v_jax, causal=True)

    _jax_close(out_jax.values, out_np.values, atol=1e-4)


def test_varlen_layernorm_jax_matches_numpy():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 8)).astype(np.float32) for n in [3, 5, 2]]

    out_np = varlen_layernorm(scree.pack(seqs), eps=1e-5)
    out_jax = varlen_layernorm(scree.pack([jnp.array(s) for s in seqs]), eps=1e-5)
    _jax_close(out_jax.values, out_np.values, atol=1e-4)


def test_varlen_rmsnorm_jax_matches_numpy():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 8)).astype(np.float32) for n in [3, 5, 2]]

    out_np = varlen_rmsnorm(scree.pack(seqs), eps=1e-6)
    out_jax = varlen_rmsnorm(scree.pack([jnp.array(s) for s in seqs]), eps=1e-6)
    _jax_close(out_jax.values, out_np.values, atol=1e-4)


def test_varlen_softmax_jax_matches_numpy():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 3)).astype(np.float32) for n in [3, 5, 2]]

    out_np = varlen_softmax(scree.pack(seqs))
    out_jax = varlen_softmax(scree.pack([jnp.array(s) for s in seqs]))
    _jax_close(out_jax.values, out_np.values, atol=1e-5)
