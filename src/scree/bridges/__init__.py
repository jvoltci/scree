"""Bridges between scree.Array and existing ecosystem objects.

A scree.Array is the canonical primitive; the bridges make migration
from existing varlen representations one-line. Each bridge is zero-copy
where the underlying memory layout allows.

Bridges shipped in v0.0:
- ``to_torch_nested`` / ``from_torch_nested`` — PyTorch jagged NestedTensor
- ``to_hf_padded`` / ``from_hf_padded`` — HuggingFace attention_mask convention
- ``to_torch`` / ``to_numpy`` — cross-framework via DLPack

FlashAttention's ``cu_seqlens`` convention is handled directly by
``scree.from_cu_seqlens`` since the offsets format is identical.
"""

from ._torch_nested import to_torch_nested, from_torch_nested
from ._hf_padded import to_hf_padded, from_hf_padded
from ._dlpack import to_torch, to_numpy

__all__ = [
    "to_torch_nested",
    "from_torch_nested",
    "to_hf_padded",
    "from_hf_padded",
    "to_torch",
    "to_numpy",
]
