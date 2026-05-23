"""A single training step on a scree-native transformer block.

Demonstrates that PyTorch autograd flows cleanly through scree.Array —
the wrapper is transparent to backprop. We build the same pre-norm
transformer block as example 02, register the weights as
``torch.nn.Parameter``, run forward + loss + backward + an Adam step,
and watch the loss decrease over a few iterations on a synthetic
copy-task.

This is not a benchmark — it's a proof that scree composes with normal
PyTorch training infrastructure.

Run:
    python examples/03_train_step.py
"""

from __future__ import annotations

import math

import torch

import scree
from scree.kernels.reference import varlen_attention, varlen_rmsnorm


def transformer_block(
    x: scree.Array,
    Wqkv: torch.Tensor,
    Wo: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    n_heads: int,
) -> scree.Array:
    """Pre-norm transformer block. Same shape as example 02 but on torch tensors."""
    model_dim = x.values.shape[-1]
    head_dim = model_dim // n_heads

    h = varlen_rmsnorm(x)
    qkv = (h.values @ Wqkv).reshape(-1, 3, n_heads, head_dim)
    q = scree.Array(values=qkv[:, 0], offsets=x.offsets)
    k = scree.Array(values=qkv[:, 1], offsets=x.offsets)
    v = scree.Array(values=qkv[:, 2], offsets=x.offsets)
    attn_out = varlen_attention(q, k, v, causal=True)
    attn_proj = attn_out.values.reshape(-1, model_dim) @ Wo
    h1 = scree.Array(values=x.values + attn_proj, offsets=x.offsets)

    ff_in = varlen_rmsnorm(h1)
    h_ff = torch.nn.functional.gelu(ff_in.values @ W1)
    ff_out = h_ff @ W2
    return scree.Array(values=h1.values + ff_out, offsets=x.offsets)


def main() -> None:
    torch.manual_seed(0)

    # Synthetic copy task: input == target, mean-squared loss.
    # Three sequences of varied length, model dim 32, 4 heads.
    lengths = [6, 4, 8]
    model_dim, n_heads, ff_dim = 32, 4, 64

    seqs = [torch.randn(n, model_dim) for n in lengths]
    x = scree.pack(seqs)
    target = x.values.clone().detach()  # input is the target

    scale = 1.0 / math.sqrt(model_dim)
    Wqkv = torch.nn.Parameter(torch.randn(model_dim, 3 * model_dim) * scale)
    Wo = torch.nn.Parameter(torch.randn(model_dim, model_dim) * scale)
    W1 = torch.nn.Parameter(torch.randn(model_dim, ff_dim) * scale)
    W2 = torch.nn.Parameter(torch.randn(ff_dim, model_dim) * (1.0 / math.sqrt(ff_dim)))

    optimizer = torch.optim.Adam([Wqkv, Wo, W1, W2], lr=3e-3)

    print(f"task: copy {sum(lengths)} packed tokens, model_dim={model_dim}, n_heads={n_heads}")
    print()
    losses = []
    for step in range(30):
        optimizer.zero_grad()
        out = transformer_block(x, Wqkv, Wo, W1, W2, n_heads=n_heads)
        loss = ((out.values - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == 29:
            print(f"  step {step:3d}: loss = {loss.item():.6f}")

    assert losses[-1] < losses[0], "training did not reduce loss — autograd may be broken"
    print()
    print(f"loss reduced {losses[0]:.4f} → {losses[-1]:.6f}  ({losses[0] / losses[-1]:.1f}× lower)")
    print("autograd flows through scree.Array correctly.")


if __name__ == "__main__":
    main()
