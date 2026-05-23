"""Core scree.Array type and construction operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def _is_mlx(x: Any) -> bool:
    return type(x).__module__.startswith("mlx")


@dataclass(frozen=True)
class Array:
    """A packed values+offsets array with one variable-length dimension.

    Variable-length sequences stored as a flat values buffer plus offsets
    pointing at row boundaries.

    Example
    -------
    Three sequences of lengths [4, 2, 5], each with feature dim 8:

        values: shape (11, 8)   # 4+2+5 along ragged_dim=0
        offsets: [0, 4, 6, 11]  # length B+1
        ragged_dim: 0

    Construct with ``scree.pack([seq1, seq2, seq3])``.
    """

    values: Any
    offsets: Any
    ragged_dim: int = 0

    def __post_init__(self) -> None:
        if self.values.ndim < 1:
            raise ValueError(f"values must be at least 1-D, got {self.values.ndim}-D")
        if not (0 <= self.ragged_dim < self.values.ndim):
            raise ValueError(
                f"ragged_dim={self.ragged_dim} out of range for {self.values.ndim}-D values"
            )
        if self.offsets.ndim != 1:
            raise ValueError(f"offsets must be 1-D, got {self.offsets.ndim}-D")
        if len(self.offsets) < 2:
            raise ValueError("offsets must have length >= 2")
        if int(self.offsets[0]) != 0:
            raise ValueError(f"offsets[0] must be 0, got {int(self.offsets[0])}")
        ragged_size = self.values.shape[self.ragged_dim]
        if int(self.offsets[-1]) != ragged_size:
            raise ValueError(
                f"offsets[-1] ({int(self.offsets[-1])}) must equal "
                f"values.shape[ragged_dim={self.ragged_dim}] ({ragged_size})"
            )

    @property
    def batch_size(self) -> int:
        return len(self.offsets) - 1

    @property
    def lengths(self) -> Any:
        """Per-row lengths: lengths[i] = offsets[i+1] - offsets[i]."""
        return self.offsets[1:] - self.offsets[:-1]

    @property
    def total_length(self) -> int:
        return int(self.offsets[-1])

    @property
    def dtype(self) -> Any:
        return self.values.dtype

    @property
    def feature_shape(self) -> tuple:
        return tuple(s for i, s in enumerate(self.values.shape) if i != self.ragged_dim)

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self) -> str:
        return (
            f"scree.Array(batch_size={self.batch_size}, "
            f"total_length={self.total_length}, "
            f"feature_shape={self.feature_shape}, "
            f"dtype={self.dtype})"
        )


def pack(arrays: List[Any], ragged_dim: int = 0) -> Array:
    """Pack a list of arrays into a single scree.Array.

    All arrays must share dtype and all non-ragged dims. The first array
    determines the backend (numpy or torch).
    """
    if not arrays:
        raise ValueError("Cannot pack an empty list")
    first = arrays[0]
    lengths = [a.shape[ragged_dim] for a in arrays]

    if _is_torch(first):
        import torch

        offsets = torch.zeros(len(arrays) + 1, dtype=torch.int32, device=first.device)
        offsets[1:] = torch.tensor(lengths, dtype=torch.int32, device=first.device).cumsum(0)
        values = torch.cat(arrays, dim=ragged_dim)
    elif _is_mlx(first):
        import mlx.core as mx

        cumsum = 0
        offsets_list = [0]
        for length in lengths:
            cumsum += length
            offsets_list.append(cumsum)
        offsets = mx.array(offsets_list, dtype=mx.int32)
        values = mx.concatenate(arrays, axis=ragged_dim)
    else:
        import numpy as np

        offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(lengths)
        values = np.concatenate(arrays, axis=ragged_dim)

    return Array(values=values, offsets=offsets, ragged_dim=ragged_dim)


def unpack(arr: Array) -> List[Any]:
    """Unpack a scree.Array into a list of arrays.

    Returned slices are views into the original ``values`` where possible.
    """
    out: List[Any] = []
    rd = arr.ragged_dim
    ndim = arr.values.ndim
    for i in range(arr.batch_size):
        start = int(arr.offsets[i])
        end = int(arr.offsets[i + 1])
        slc = [slice(None)] * ndim
        slc[rd] = slice(start, end)
        out.append(arr.values[tuple(slc)])
    return out


def to_padded(arr: Array, side: str = "right", fill_value: float = 0.0) -> Tuple[Any, Any]:
    """Convert a scree.Array to a padded dense array + mask.

    Returns ``(padded, mask)`` where:
    - ``padded.shape == (batch_size, max_len, *feature_dims)``
    - ``mask.shape == (batch_size, max_len)`` — True for valid positions
    """
    if arr.ragged_dim != 0:
        raise NotImplementedError("to_padded supports ragged_dim=0 only in v0.1")
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")

    lengths = [int(arr.offsets[i + 1] - arr.offsets[i]) for i in range(arr.batch_size)]
    max_len = max(lengths) if lengths else 0
    feature_shape = arr.values.shape[1:]
    batch = arr.batch_size

    if _is_mlx(arr.values):
        # MLX prefers a mutation-free construction (lazy graph).
        import mlx.core as mx

        rows_padded = []
        rows_mask = []
        for i, length in enumerate(lengths):
            start = int(arr.offsets[i])
            row = arr.values[start : start + length]
            pad_shape = (max_len - length, *feature_shape)
            pad = mx.full(pad_shape, fill_value, dtype=arr.values.dtype)
            valid_mask = mx.ones((length,), dtype=mx.bool_)
            pad_mask = mx.zeros((max_len - length,), dtype=mx.bool_)
            if side == "right":
                rows_padded.append(mx.concatenate([row, pad], axis=0))
                rows_mask.append(mx.concatenate([valid_mask, pad_mask], axis=0))
            else:
                rows_padded.append(mx.concatenate([pad, row], axis=0))
                rows_mask.append(mx.concatenate([pad_mask, valid_mask], axis=0))
        padded = mx.stack(rows_padded, axis=0)
        mask = mx.stack(rows_mask, axis=0)
        return padded, mask

    if _is_torch(arr.values):
        import torch

        padded = torch.full(
            (batch, max_len, *feature_shape),
            fill_value,
            dtype=arr.values.dtype,
            device=arr.values.device,
        )
        mask = torch.zeros((batch, max_len), dtype=torch.bool, device=arr.values.device)
    else:
        import numpy as np

        padded = np.full((batch, max_len, *feature_shape), fill_value, dtype=arr.values.dtype)
        mask = np.zeros((batch, max_len), dtype=np.bool_)

    for i, length in enumerate(lengths):
        start = int(arr.offsets[i])
        row = arr.values[start : start + length]
        if side == "right":
            padded[i, :length] = row
            mask[i, :length] = True
        else:  # left
            padded[i, max_len - length :] = row
            mask[i, max_len - length :] = True

    return padded, mask


def from_padded(padded: Any, mask: Any) -> Array:
    """Convert ``(padded, mask)`` to a scree.Array.

    Assumes right-padding (mask is True on the left side of each row).
    """
    batch = padded.shape[0]

    if _is_torch(padded):
        import torch

        lengths = [int(mask[i].sum().item()) for i in range(batch)]
        rows = [padded[i, : lengths[i]] for i in range(batch)]
        values = torch.cat(rows, dim=0)
        offsets = torch.zeros(batch + 1, dtype=torch.int32, device=padded.device)
        offsets[1:] = torch.tensor(lengths, dtype=torch.int32, device=padded.device).cumsum(0)
    elif _is_mlx(padded):
        import mlx.core as mx

        lengths = [int(mask[i].sum().item()) for i in range(batch)]
        rows = [padded[i, : lengths[i]] for i in range(batch)]
        values = mx.concatenate(rows, axis=0)
        cumsum = 0
        offsets_list = [0]
        for length in lengths:
            cumsum += length
            offsets_list.append(cumsum)
        offsets = mx.array(offsets_list, dtype=mx.int32)
    else:
        import numpy as np

        lengths = [int(mask[i].sum()) for i in range(batch)]
        rows = [padded[i, : lengths[i]] for i in range(batch)]
        values = np.concatenate(rows, axis=0)
        offsets = np.zeros(batch + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(lengths)

    return Array(values=values, offsets=offsets, ragged_dim=0)


def from_cu_seqlens(values: Any, cu_seqlens: Any) -> Array:
    """Construct a scree.Array from FlashAttention's cu_seqlens convention.

    FlashAttention's ``cu_seqlens`` is exactly scree's ``offsets``. Zero-copy.
    """
    return Array(values=values, offsets=cu_seqlens, ragged_dim=0)
