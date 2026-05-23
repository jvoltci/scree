"""Bridge between scree.Array and HuggingFace ``(hidden_states, attention_mask)``.

HF Transformers passes variable-length data as a right-padded dense tensor
plus an ``attention_mask`` of shape ``(batch, seq_len)`` where 1 marks
real tokens and 0 marks padding. This bridge converts both directions.
"""

from __future__ import annotations

from typing import Any, Tuple

from .._core import Array, _is_torch, from_padded, to_padded


def from_hf_padded(hidden_states: Any, attention_mask: Any) -> Array:
    """Convert HF ``(hidden_states, attention_mask)`` to a scree.Array.

    Parameters
    ----------
    hidden_states : array-like, shape (batch, seq_len, *features)
    attention_mask : array-like, shape (batch, seq_len)
        1 for real tokens, 0 for padding (HF convention).
    """
    if _is_torch(attention_mask):
        mask = attention_mask.to(dtype=__import__("torch").bool)
    else:
        mask = attention_mask.astype(bool)
    return from_padded(hidden_states, mask)


def to_hf_padded(arr: Array) -> Tuple[Any, Any]:
    """Convert a scree.Array to HF ``(hidden_states, attention_mask)``.

    Returns ``(hidden_states, attention_mask)`` where attention_mask is
    int64 with 1 for valid positions, 0 for padding (HF convention).
    """
    padded, mask = to_padded(arr, side="right")
    if _is_torch(mask):
        import torch

        mask = mask.to(dtype=torch.int64)
    else:
        import numpy as np

        mask = mask.astype(np.int64)
    return padded, mask
