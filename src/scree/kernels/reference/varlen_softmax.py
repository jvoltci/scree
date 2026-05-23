"""Reference (slow but correct) implementation of varlen softmax.

Softmax along the ragged dimension. Unlike layernorm, this is non-trivial
for packed data because softmax must be computed within each sequence
separately — not across the full concatenated buffer.
"""

from __future__ import annotations

from ..._core import Array, _is_torch


def varlen_softmax(arr: Array) -> Array:
    """Softmax along the ragged dimension, per-sequence.

    Each row (sequence) is softmaxed independently. The output has the
    same shape and offsets as the input.
    """
    if arr.ragged_dim != 0:
        raise NotImplementedError("varlen_softmax supports ragged_dim=0 only in v0.1")

    if _is_torch(arr.values):
        import torch

        out_rows = []
        for i in range(arr.batch_size):
            s = int(arr.offsets[i])
            e = int(arr.offsets[i + 1])
            out_rows.append(torch.softmax(arr.values[s:e], dim=0))
        values = torch.cat(out_rows, dim=0)
    else:
        import numpy as np

        out_rows = []
        for i in range(arr.batch_size):
            s = int(arr.offsets[i])
            e = int(arr.offsets[i + 1])
            row = arr.values[s:e]
            row_max = row.max(axis=0, keepdims=True)
            row_exp = np.exp(row - row_max)
            out_rows.append(row_exp / row_exp.sum(axis=0, keepdims=True))
        values = np.concatenate(out_rows, axis=0)

    return Array(values=values, offsets=arr.offsets, ragged_dim=0)
