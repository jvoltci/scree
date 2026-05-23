"""Bridge between scree.Array and torch.nested (jagged layout)."""

from __future__ import annotations

from typing import Any

from .._core import Array, unpack


def to_torch_nested(arr: Array) -> Any:
    """Convert a scree.Array to a torch.NestedTensor (jagged layout).

    The conversion materializes per-row views from the packed buffer
    and hands them to ``torch.nested.nested_tensor``. Internally torch
    may share the underlying storage; we don't promise zero-copy.
    """
    import torch

    if arr.ragged_dim != 0:
        raise NotImplementedError("to_torch_nested supports ragged_dim=0 only")
    rows = unpack(arr)
    return torch.nested.nested_tensor(list(rows), layout=torch.jagged)


def from_torch_nested(nt: Any) -> Array:
    """Convert a torch.nested.NestedTensor (jagged) to a scree.Array.

    Uses the jagged NestedTensor's underlying values + offsets directly
    (zero-copy when supported by the torch version).
    """
    if not getattr(nt, "is_nested", False):
        raise TypeError(f"expected a NestedTensor, got {type(nt).__name__}")
    import torch

    # torch.jagged NT exposes .values() and .offsets() on modern torch versions.
    # Fall back to unbinding + repacking if those aren't available.
    try:
        values = nt.values()
        offsets = nt.offsets().to(torch.int32)
    except (AttributeError, RuntimeError):
        rows = list(nt.unbind())
        values = torch.cat(rows, dim=0)
        lengths = [r.shape[0] for r in rows]
        offsets = torch.zeros(len(rows) + 1, dtype=torch.int32, device=values.device)
        offsets[1:] = torch.tensor(lengths, dtype=torch.int32, device=values.device).cumsum(0)

    return Array(values=values, offsets=offsets, ragged_dim=0)
