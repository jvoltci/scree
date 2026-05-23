"""HuggingFace Transformers compatibility recipe.

Demonstrates the migration pattern for code that already uses the HF
``(hidden_states, attention_mask)`` convention:

    1. Receive HF-style padded inputs at the boundary.
    2. Convert to scree.Array — drops padding tokens.
    3. Run scree-native operations on the packed buffer.
    4. Convert back to HF padded form for downstream code that expects it.
    5. Verify numerical equivalence with the original padded computation.

This example does NOT require ``transformers`` installed — it simulates
the HF interface (a padded tensor + an int64 0/1 mask) which is enough
to demonstrate the migration pattern any HF user follows.

Run:
    python examples/04_hf_compat.py
"""

from __future__ import annotations

import numpy as np
import torch

import scree
import scree.bridges as bridges
from scree.kernels.reference import varlen_layernorm


def simulate_hf_output(batch_size: int, max_seq: int, feature_dim: int, lengths: list[int]):
    """Produce (hidden_states, attention_mask) like an HF model would."""
    torch.manual_seed(0)
    hidden_states = torch.zeros(batch_size, max_seq, feature_dim)
    attention_mask = torch.zeros(batch_size, max_seq, dtype=torch.int64)
    for i, length in enumerate(lengths):
        hidden_states[i, :length] = torch.randn(length, feature_dim)
        attention_mask[i, :length] = 1
    return hidden_states, attention_mask


def hf_native_layernorm(hidden_states: torch.Tensor, attention_mask: torch.Tensor, eps: float = 1e-5):
    """A 'reference' HF-style layernorm computed per-row, applied through the mask.

    The kind of code you'd write today before adopting scree. Mask is
    applied AFTER the normalization so padding positions stay at 0 — but
    the normalization itself processes all positions including padding.
    """
    x = hidden_states
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    y = (x - mean) / torch.sqrt(var + eps)
    return y * attention_mask.unsqueeze(-1).to(y.dtype)


def main() -> None:
    # Simulate HF output: batch 4, max_seq 16, hidden dim 32, with three real lengths.
    lengths = [6, 12, 4, 8]
    batch, max_seq, feature_dim = 4, max(lengths), 32

    hidden_states, attention_mask = simulate_hf_output(batch, max_seq, feature_dim, lengths)

    print(f"hf output:")
    print(f"  hidden_states.shape    = {tuple(hidden_states.shape)}")
    print(f"  attention_mask.shape   = {tuple(attention_mask.shape)}")
    print(f"  padded tokens:           {batch * max_seq}")
    print(f"  real tokens:             {sum(lengths)}")
    print(f"  padding waste:           {(batch * max_seq - sum(lengths)) / (batch * max_seq):.0%}")
    print()

    # Step 1: convert to scree.Array. Padding is dropped.
    arr = bridges.from_hf_padded(hidden_states, attention_mask)
    print(f"after bridges.from_hf_padded:")
    print(f"  scree.Array            = {arr}")
    print(f"  values.shape           = {tuple(arr.values.shape)}     # no padding")
    print(f"  offsets                = {arr.offsets.tolist()}")
    print()

    # Step 2: run scree-native operation on the packed buffer.
    arr_norm = varlen_layernorm(arr, eps=1e-5)
    print(f"after scree varlen_layernorm:")
    print(f"  out.values.shape       = {tuple(arr_norm.values.shape)}")
    print()

    # Step 3: convert back to HF format for downstream code that expects it.
    hidden_back, mask_back = bridges.to_hf_padded(arr_norm)
    print(f"after bridges.to_hf_padded (round-trip):")
    print(f"  hidden_back.shape      = {tuple(hidden_back.shape)}")
    print(f"  mask_back.shape        = {tuple(mask_back.shape)}")
    print()

    # Step 4: numerical equivalence with the HF-native path.
    hf_native = hf_native_layernorm(hidden_states, attention_mask)
    max_diff = (hidden_back - hf_native).abs().max().item()
    print(f"numerical check vs HF-native layernorm:")
    print(f"  max abs diff           = {max_diff:.2e}")
    assert max_diff < 1e-5, "scree and HF-native should agree to fp tolerance"
    print(f"  result: PASS — scree and HF-native produce identical output on real tokens")
    print()

    # Step 5: mask round-trip is exact (no precision loss).
    assert torch.equal(mask_back, attention_mask), "attention_mask should round-trip exactly"
    print(f"  attention_mask round-trip: exact match")


if __name__ == "__main__":
    main()
