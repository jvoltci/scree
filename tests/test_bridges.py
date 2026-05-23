"""Tests for the bridges between scree.Array and existing ecosystem objects."""

import numpy as np
import pytest

import scree
from scree.bridges import (
    from_hf_padded,
    from_torch_nested,
    to_hf_padded,
    to_numpy,
    to_torch,
    to_torch_nested,
)


def test_hf_padded_roundtrip_numpy():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 4)).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    hidden, mask = to_hf_padded(arr)
    # HF uses int mask with 1/0, not bool
    assert mask.dtype == np.int64
    assert mask.sum() == arr.total_length
    arr2 = from_hf_padded(hidden, mask)
    np.testing.assert_array_equal(arr.values, arr2.values)
    np.testing.assert_array_equal(arr.offsets, arr2.offsets)


def test_hf_padded_mask_convention():
    """attention_mask must be 1 for real tokens, 0 for pad — HF's convention."""
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 4)).astype(np.float32) for n in [2, 5]]
    arr = scree.pack(seqs)
    _, mask = to_hf_padded(arr)
    # Row 0 has length 2, padded to length 5 → [1, 1, 0, 0, 0]
    assert mask[0].tolist() == [1, 1, 0, 0, 0]
    assert mask[1].tolist() == [1, 1, 1, 1, 1]


def test_to_numpy_passthrough():
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 4)).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    arr2 = to_numpy(arr)
    # numpy input → numpy output, same object expected
    assert arr2 is arr


def test_to_torch_from_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 4)).astype(np.float32) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    arr_t = to_torch(arr)
    assert torch.is_tensor(arr_t.values)
    assert torch.is_tensor(arr_t.offsets)
    assert arr_t.batch_size == 3
    assert arr_t.total_length == 10
    # Values should match
    np.testing.assert_array_equal(arr_t.values.numpy(), arr.values)


def test_to_numpy_from_torch():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    seqs = [torch.randn(n, 4, generator=gen) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    arr_np = to_numpy(arr)
    assert isinstance(arr_np.values, np.ndarray)
    assert isinstance(arr_np.offsets, np.ndarray)
    np.testing.assert_allclose(arr_np.values, arr.values.numpy())


def test_torch_nested_roundtrip():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    seqs = [torch.randn(n, 4, generator=gen) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    nt = to_torch_nested(arr)
    assert nt.is_nested
    arr2 = from_torch_nested(nt)
    torch.testing.assert_close(arr.values, arr2.values)
    torch.testing.assert_close(arr.offsets, arr2.offsets.to(arr.offsets.dtype))


def test_hf_padded_torch():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    seqs = [torch.randn(n, 4, generator=gen) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    hidden, mask = to_hf_padded(arr)
    assert mask.dtype == torch.int64
    arr2 = from_hf_padded(hidden, mask)
    torch.testing.assert_close(arr.values, arr2.values)
