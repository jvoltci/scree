"""Cross-framework bridges via DLPack (zero-copy where memory layout allows)."""

from __future__ import annotations

from typing import Any

from .._core import Array, _is_torch


def to_torch(arr: Array) -> Array:
    """Re-export a scree.Array with its values/offsets as torch tensors.

    Zero-copy on CPU via ``torch.from_numpy``; zero-copy on GPU via DLPack.
    """
    import torch

    if _is_torch(arr.values):
        return arr

    # numpy -> torch
    values = torch.from_numpy(arr.values) if hasattr(arr.values, "__array__") else _via_dlpack(arr.values, torch)
    offsets = torch.from_numpy(arr.offsets) if hasattr(arr.offsets, "__array__") else _via_dlpack(arr.offsets, torch)
    return Array(values=values, offsets=offsets.to(torch.int32), ragged_dim=arr.ragged_dim)


def to_numpy(arr: Array) -> Array:
    """Re-export a scree.Array with its values/offsets as numpy arrays.

    Zero-copy from CPU torch tensors; for GPU torch tensors, copies to host.
    """
    import numpy as np

    if not _is_torch(arr.values):
        return arr

    # torch -> numpy
    values = arr.values.detach().cpu().numpy()
    offsets = arr.offsets.detach().cpu().numpy()
    return Array(values=values, offsets=offsets.astype(np.int32), ragged_dim=arr.ragged_dim)


def _via_dlpack(x: Any, target_module: Any) -> Any:
    """Reimport ``x`` into ``target_module`` via the DLPack protocol."""
    return target_module.from_dlpack(x)
