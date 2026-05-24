"""Array API conformance tests for scree.Array dense dims.

scree.Array exposes a flat ``values`` buffer plus ``offsets``; the
``values`` field is a regular array of whatever backend was passed in
(NumPy / PyTorch / MLX / JAX), and operations on the *dense* dimensions
of ``values`` should compose with the Python Array API standard ops
that those backends already support.

These tests verify the subset of the Array API spec that's relevant to
scree's contract:
  - dtype attribute exists and matches expectations
  - shape attribute exists and is a tuple of ints
  - ndim attribute is consistent with shape
  - __getitem__ slicing works on the ragged dim and produces correct shapes
  - basic elementwise ops on values produce arrays that can be wrapped
    back into a new scree.Array with the same offsets

Notes
-----
scree itself is intentionally NOT registered as an Array API namespace —
it's a typed wrapper, not a math library. The Array API compliance lives
in the *underlying* values tensor.
"""

import numpy as np
import pytest

import scree


def test_values_has_array_api_attributes_numpy():
    """values: numpy ndarray should have dtype, shape, ndim attributes."""
    seqs = [np.random.randn(n, 4).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    assert hasattr(arr.values, "dtype")
    assert hasattr(arr.values, "shape")
    assert hasattr(arr.values, "ndim")
    assert isinstance(arr.values.shape, tuple)
    assert all(isinstance(d, int) for d in arr.values.shape)
    assert arr.values.ndim == 2


def test_values_has_array_api_attributes_torch():
    torch = pytest.importorskip("torch")
    seqs = [torch.randn(n, 4) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    assert hasattr(arr.values, "dtype")
    assert hasattr(arr.values, "shape")
    assert hasattr(arr.values, "ndim")


def test_slicing_along_ragged_dim_produces_correct_shapes():
    """arr.values[start:end] over a sequence's offset range gives that row."""
    seqs = [np.random.randn(n, 4).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    lengths = arr.lengths.tolist()
    starts = arr.offsets[:-1].tolist()
    for i, (start, length) in enumerate(zip(starts, lengths)):
        row = arr.values[start : start + length]
        assert row.shape == (length, 4)
        np.testing.assert_array_equal(row, seqs[i])


def test_elementwise_ops_preserve_scree_array():
    """Wrapping an elementwise-op output back as scree.Array preserves invariants."""
    seqs = [np.random.randn(n, 4).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)

    # Scale by 2.0 — pure elementwise
    arr2 = scree.Array(values=arr.values * 2.0, offsets=arr.offsets, ragged_dim=0)
    assert arr2.batch_size == arr.batch_size
    assert arr2.total_length == arr.total_length
    assert arr2.feature_shape == arr.feature_shape
    np.testing.assert_allclose(arr2.values, arr.values * 2.0)


def test_dlpack_protocol_present_on_values():
    """torch values should support __dlpack__; numpy values should too on >=1.22."""
    torch = pytest.importorskip("torch")
    seqs = [torch.randn(n, 4) for n in [3, 5]]
    arr = scree.pack(seqs)
    assert hasattr(arr.values, "__dlpack__")


def test_dtype_consistency_after_pack():
    """All input arrays share dtype, so the packed values has that dtype."""
    for dtype in (np.float32, np.float64, np.int32):
        seqs = [np.zeros((n, 4), dtype=dtype) for n in [3, 5]]
        arr = scree.pack(seqs)
        assert arr.values.dtype == dtype
        assert arr.dtype == dtype


def test_arr_dtype_matches_values_dtype():
    seqs = [np.random.randn(n, 4).astype(np.float32) for n in [3, 5]]
    arr = scree.pack(seqs)
    assert arr.dtype is arr.values.dtype


def test_feature_shape_excludes_ragged_dim():
    """For a 3-D values shape (total, A, B) with ragged_dim=0, feature_shape == (A, B)."""
    seqs = [np.random.randn(n, 4, 8).astype(np.float32) for n in [3, 5]]
    arr = scree.pack(seqs)
    assert arr.values.ndim == 3
    assert arr.feature_shape == (4, 8)


def test_array_api_concat_round_trip_numpy():
    """Build an Array via pack(), then concatenate values back to verify Array API concat semantics."""
    seqs = [np.random.randn(n, 4).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    # Manual reconstruction via np.concatenate (Array API standard op).
    manual = np.concatenate(seqs, axis=0)
    np.testing.assert_array_equal(arr.values, manual)


def test_array_api_concat_round_trip_torch():
    torch = pytest.importorskip("torch")
    seqs = [torch.randn(n, 4) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    manual = torch.cat(seqs, dim=0)  # Array API standard op
    torch.testing.assert_close(arr.values, manual)
