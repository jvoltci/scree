"""Property-based tests for scree invariants and operations.

These tests use hypothesis to generate a wide range of inputs and verify
properties that should hold for ALL valid scree.Arrays — catching edge
cases the targeted unit tests would miss.

Tests run across all available backends (NumPy + PyTorch + MLX + JAX);
unavailable backends are skipped via pytest.importorskip on the test
level (not the file level — we don't want to skip NumPy tests just
because Torch is missing).

Properties tested:
- pack/unpack roundtrip is identity
- to_padded/from_padded roundtrip is identity
- Array invariants: offsets[0]==0, offsets[-1]==values.shape[ragged_dim]
- Derived properties: lengths.sum() == total_length, batch_size == len(lengths)
- varlen_softmax produces rows summing to 1
- varlen_attention preserves input shape
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import scree
from scree.kernels.reference import (
    varlen_attention,
    varlen_layernorm,
    varlen_rmsnorm,
    varlen_softmax,
)


def _available_backends() -> list[str]:
    """Detect which optional backends can be tested in this environment."""
    backends = ["numpy"]
    for name, mod in [("torch", "torch"), ("mlx", "mlx.core"), ("jax", "jax.numpy")]:
        try:
            importlib.import_module(mod)
            backends.append(name)
        except ImportError:
            pass
    return backends


BACKENDS = _available_backends()


def _convert_arrays(numpy_arrays: list[np.ndarray], backend: str) -> list:
    """Cast a list of numpy arrays to the target backend."""
    if backend == "numpy":
        return numpy_arrays
    if backend == "torch":
        import torch

        return [torch.from_numpy(a) for a in numpy_arrays]
    if backend == "mlx":
        import mlx.core as mx

        return [mx.array(a) for a in numpy_arrays]
    if backend == "jax":
        import jax.numpy as jnp

        return [jnp.array(a) for a in numpy_arrays]
    raise ValueError(f"unknown backend: {backend!r}")


def _to_numpy(arr) -> np.ndarray:
    """Cast an arbitrary-backend array to a NumPy array for comparison."""
    return np.array(arr)


# Hypothesis strategies for generating valid scree-shaped inputs.

# A list of 1-32 positive lengths.
lengths_strategy = st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=32)

# A modest feature dim — keeps tests fast.
feature_dim_strategy = st.integers(min_value=1, max_value=16)


@st.composite
def packed_arrays(draw, lengths=lengths_strategy, feature_dim=feature_dim_strategy):
    """Generate a list of variable-length 2-D arrays for scree.pack."""
    lens = draw(lengths)
    d = draw(feature_dim)
    rng = np.random.default_rng(0)
    return [rng.standard_normal((n, d)).astype(np.float32) for n in lens]


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays())
def test_pack_unpack_roundtrip(seqs):
    """For any list of arrays sharing dtype + non-ragged dims, pack+unpack == identity."""
    arr = scree.pack(seqs)
    assert arr.batch_size == len(seqs)
    assert arr.total_length == sum(s.shape[0] for s in seqs)
    out = scree.unpack(arr)
    assert len(out) == len(seqs)
    for original, recovered in zip(seqs, out):
        np.testing.assert_array_equal(original, recovered)


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays())
def test_to_from_padded_roundtrip(seqs):
    """For any scree.Array, to_padded then from_padded == identity."""
    arr = scree.pack(seqs)
    padded, mask = scree.to_padded(arr)
    assert padded.shape[0] == arr.batch_size
    assert mask.sum() == arr.total_length
    arr2 = scree.from_padded(padded, mask)
    np.testing.assert_array_equal(arr.values, arr2.values)
    np.testing.assert_array_equal(arr.offsets, arr2.offsets)


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays())
def test_array_invariants_hold(seqs):
    """offsets[0]==0, offsets[-1]==total_length, monotonic increasing."""
    arr = scree.pack(seqs)
    assert int(arr.offsets[0]) == 0
    assert int(arr.offsets[-1]) == arr.total_length
    # Monotonic non-decreasing
    diffs = np.diff(arr.offsets)
    assert (diffs >= 0).all()
    # lengths agree with input
    assert arr.lengths.tolist() == [s.shape[0] for s in seqs]


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays())
def test_batch_size_and_total_length_consistent(seqs):
    """batch_size == len(offsets) - 1, total_length == sum(lengths)."""
    arr = scree.pack(seqs)
    assert arr.batch_size == len(arr.offsets) - 1
    assert arr.total_length == int(arr.lengths.sum())
    assert arr.total_length == arr.values.shape[arr.ragged_dim]


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays(feature_dim=st.integers(min_value=2, max_value=8)))
def test_varlen_softmax_rows_sum_to_one(seqs):
    """Per-sequence softmax: each row's softmax output sums to 1.0."""
    arr = scree.pack(seqs)
    out = varlen_softmax(arr)
    for row in scree.unpack(out):
        # Sum over the ragged dim (axis 0); should be feature_dim-shaped vector of 1.0s
        row_sum = row.sum(axis=0)
        np.testing.assert_allclose(row_sum, np.ones_like(row_sum), atol=1e-5)


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays())
def test_varlen_layernorm_preserves_shape(seqs):
    """LayerNorm output has the same shape and offsets as input."""
    arr = scree.pack(seqs)
    out = varlen_layernorm(arr, eps=1e-5)
    assert out.values.shape == arr.values.shape
    np.testing.assert_array_equal(out.offsets, arr.offsets)


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays(feature_dim=st.integers(min_value=4, max_value=16)))
def test_varlen_layernorm_zero_mean_unit_var(seqs):
    """Without weight/bias, LayerNorm output has approx zero mean and unit variance
    along the last dim."""
    arr = scree.pack(seqs)
    out = varlen_layernorm(arr, eps=1e-5)
    mean = out.values.mean(axis=-1)
    var = out.values.var(axis=-1)
    np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-5)
    # Variance should be close to 1 (slightly less due to eps in denominator)
    assert (var > 0.9).all() and (var <= 1.0 + 1e-5).all()


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays(feature_dim=st.integers(min_value=4, max_value=16)))
def test_varlen_rmsnorm_unit_rms(seqs):
    """Without weight, RMSNorm output has approx unit RMS along the last dim."""
    arr = scree.pack(seqs)
    out = varlen_rmsnorm(arr, eps=1e-6)
    rms = np.sqrt((out.values * out.values).mean(axis=-1))
    np.testing.assert_allclose(rms, np.ones_like(rms), atol=1e-3)


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=None)
@given(lengths=lengths_strategy.map(lambda l: l[:6]))  # cap batch at 6 for attention speed
def test_varlen_attention_preserves_shape(lengths):
    """varlen_attention output has identical shape and offsets to q."""
    assume(len(lengths) > 0)
    rng = np.random.default_rng(0)
    n_heads, head_dim = 2, 4
    q_seqs = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    k_seqs = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    v_seqs = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    q, k, v = scree.pack(q_seqs), scree.pack(k_seqs), scree.pack(v_seqs)
    out = varlen_attention(q, k, v, causal=True)
    assert out.values.shape == q.values.shape
    np.testing.assert_array_equal(out.offsets, q.offsets)


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays())
def test_from_cu_seqlens_zero_copy(seqs):
    """from_cu_seqlens produces an Array sharing memory with the inputs."""
    arr = scree.pack(seqs)
    arr2 = scree.from_cu_seqlens(arr.values, arr.offsets)
    assert arr2.values is arr.values
    assert arr2.offsets is arr.offsets


@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(packed_arrays())
def test_to_padded_mask_matches_lengths(seqs):
    """Mask sum equals total_length; per-row mask sum equals row length."""
    arr = scree.pack(seqs)
    _, mask = scree.to_padded(arr)
    assert int(mask.sum()) == arr.total_length
    for i, length in enumerate(arr.lengths.tolist()):
        assert int(mask[i].sum()) == length


# Cross-backend property tests — each hypothesis case is verified across
# every available backend (NumPy + PyTorch + MLX + JAX). A bug in any
# backend fails the test for that hypothesis input.

_CROSS_BACKEND_SETTINGS = settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture, HealthCheck.large_base_example],
    deadline=None,  # cross-backend runs ~4x the work; default 200ms is too tight
    max_examples=25,  # reduce search space — we only need a few cases per backend
    print_blob=False,
)


@_CROSS_BACKEND_SETTINGS
@given(packed_arrays(lengths=st.lists(st.integers(min_value=1, max_value=16), min_size=1, max_size=8),
                     feature_dim=st.integers(min_value=1, max_value=8)))
def test_pack_unpack_roundtrip_all_backends(seqs):
    """pack/unpack roundtrip identity must hold on every backend."""
    for backend in BACKENDS:
        arrays = _convert_arrays(seqs, backend)
        arr = scree.pack(arrays)
        assert arr.batch_size == len(arrays), f"backend {backend}"
        out = scree.unpack(arr)
        assert len(out) == len(arrays), f"backend {backend}"
        for original_np, recovered in zip(seqs, out):
            np.testing.assert_array_equal(
                original_np, _to_numpy(recovered),
                err_msg=f"backend {backend}",
            )


@_CROSS_BACKEND_SETTINGS
@given(packed_arrays(lengths=st.lists(st.integers(min_value=1, max_value=16), min_size=1, max_size=8),
                     feature_dim=st.integers(min_value=1, max_value=8)))
def test_array_invariants_hold_all_backends(seqs):
    """offsets[0]==0, offsets[-1]==total_length, monotonic non-decreasing on every backend."""
    expected_lengths = [s.shape[0] for s in seqs]
    for backend in BACKENDS:
        arrays = _convert_arrays(seqs, backend)
        arr = scree.pack(arrays)
        assert int(arr.offsets[0]) == 0, f"backend {backend}"
        assert int(arr.offsets[-1]) == arr.total_length, f"backend {backend}"
        diffs = _to_numpy(arr.lengths)
        assert (diffs >= 0).all(), f"backend {backend}"
        actual_lengths = _to_numpy(arr.lengths).tolist()
        assert actual_lengths == expected_lengths, f"backend {backend}"
