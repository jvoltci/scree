"""Tests for the core scree.Array type and pack/unpack/padded ops."""

import numpy as np
import pytest

import scree


def test_array_basic_invariants_numpy():
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    offsets = np.array([0, 3, 7, 10], dtype=np.int32)
    arr = scree.Array(values=values, offsets=offsets, ragged_dim=0)
    assert arr.batch_size == 3
    assert arr.total_length == 10
    assert arr.dtype == np.float32
    assert arr.feature_shape == (2,)
    assert len(arr) == 3


def test_array_rejects_bad_offsets():
    values = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"offsets\[0\]"):
        scree.Array(values=values, offsets=np.array([1, 5, 10], dtype=np.int32))
    with pytest.raises(ValueError, match=r"offsets\[-1\]"):
        scree.Array(values=values, offsets=np.array([0, 5, 8], dtype=np.int32))


def test_array_rejects_bad_offsets_ndim():
    values = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"offsets must be 1-D"):
        scree.Array(values=values, offsets=np.array([[0, 5, 10]], dtype=np.int32))


def test_array_rejects_bad_ragged_dim():
    values = np.zeros((10, 2), dtype=np.float32)
    offsets = np.array([0, 5, 10], dtype=np.int32)
    with pytest.raises(ValueError, match=r"ragged_dim"):
        scree.Array(values=values, offsets=offsets, ragged_dim=5)


def test_pack_unpack_roundtrip_numpy():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 4)).astype(np.float32) for n in [3, 5, 2, 7]]
    arr = scree.pack(seqs)
    assert arr.batch_size == 4
    assert arr.total_length == 17
    out = scree.unpack(arr)
    assert len(out) == 4
    for a, b in zip(seqs, out):
        np.testing.assert_array_equal(a, b)


def test_pack_empty_list_raises():
    with pytest.raises(ValueError, match=r"empty"):
        scree.pack([])


def test_to_from_padded_roundtrip_numpy():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 4)).astype(np.float32) for n in [3, 5, 2, 7]]
    arr = scree.pack(seqs)
    padded, mask = scree.to_padded(arr)
    assert padded.shape == (4, 7, 4)
    assert mask.shape == (4, 7)
    assert mask.sum() == 17
    arr2 = scree.from_padded(padded, mask)
    np.testing.assert_array_equal(arr.values, arr2.values)
    np.testing.assert_array_equal(arr.offsets, arr2.offsets)


def test_to_padded_left_side():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 4)).astype(np.float32) for n in [3, 5]]
    arr = scree.pack(seqs)
    padded, mask = scree.to_padded(arr, side="left")
    # Row 0 (len 3) padded on the left of width 5
    assert mask[0].tolist() == [False, False, True, True, True]
    assert mask[1].tolist() == [True, True, True, True, True]


def test_from_cu_seqlens_numpy():
    values = np.zeros((10, 2), dtype=np.float32)
    cu_seqlens = np.array([0, 3, 7, 10], dtype=np.int32)
    arr = scree.from_cu_seqlens(values, cu_seqlens)
    assert arr.batch_size == 3
    assert arr.total_length == 10
    # values and offsets are passed through unchanged (zero-copy)
    assert arr.values is values
    assert arr.offsets is cu_seqlens


def test_lengths_property():
    values = np.zeros((10, 2), dtype=np.float32)
    offsets = np.array([0, 3, 7, 10], dtype=np.int32)
    arr = scree.Array(values=values, offsets=offsets)
    np.testing.assert_array_equal(arr.lengths, np.array([3, 4, 3], dtype=np.int32))


def test_repr_does_not_crash():
    values = np.zeros((10, 2), dtype=np.float32)
    offsets = np.array([0, 3, 7, 10], dtype=np.int32)
    arr = scree.Array(values=values, offsets=offsets)
    s = repr(arr)
    assert "scree.Array" in s
    assert "batch_size=3" in s


def test_pack_unpack_roundtrip_torch():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    seqs = [torch.randn(n, 4, generator=gen) for n in [3, 5, 2, 7]]
    arr = scree.pack(seqs)
    assert arr.batch_size == 4
    assert arr.total_length == 17
    out = scree.unpack(arr)
    for a, b in zip(seqs, out):
        torch.testing.assert_close(a, b)


def test_to_from_padded_roundtrip_torch():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    seqs = [torch.randn(n, 4, generator=gen) for n in [3, 5, 2, 7]]
    arr = scree.pack(seqs)
    padded, mask = scree.to_padded(arr)
    assert padded.shape == (4, 7, 4)
    assert mask.shape == (4, 7)
    arr2 = scree.from_padded(padded, mask)
    torch.testing.assert_close(arr.values, arr2.values)
    torch.testing.assert_close(arr.offsets, arr2.offsets)
