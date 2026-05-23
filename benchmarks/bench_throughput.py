"""CPU throughput benchmark: scree reference kernels vs padded baselines.

Measures wall-clock time per iteration for ``varlen_attention`` and
``varlen_rmsnorm`` operating on a scree.Array vs a naive padded tensor
where the operation processes all padded positions (the no-mask path).

Runs entirely on CPU; no GPU required. The point is to show that even
without a fast GPU kernel, scree wastes less compute than a padded
representation just by skipping the padding tokens.

For the GPU benchmark vs FlashAttention-2 see ``modal_bench.py``.

Usage
-----
    python benchmarks/bench_throughput.py
    python benchmarks/bench_throughput.py --num-seqs 32 --mean-len 256
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import scree
from scree.kernels.reference import varlen_attention, varlen_rmsnorm


def realistic_lengths(num_seqs: int, mean_len: int, sigma: float = 0.6, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    lengths = rng.lognormal(mean=np.log(mean_len), sigma=sigma, size=num_seqs)
    return [max(1, int(length)) for length in lengths]


def padded_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Naive padded attention — processes all positions including padding.

    This is the 'no-mask-optimization' path that scree skips by construction.
    Real HF Transformers usually masks but still pays the FLOPs of the full
    padded matmul.

    Note: this baseline produces NaN values in all-padding rows (softmax over
    all -inf is undefined). The np.errstate suppression hides the warning;
    the timing is still meaningful since the FLOPs work happens before the
    NaN materializes.
    """
    head_dim = q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.einsum("bihd,bjhd->bhij", q, k) * scale
    scores = np.where(mask[:, None, None, :], scores, -np.inf)
    scores = np.where(mask[:, None, :, None], scores, -np.inf)
    with np.errstate(invalid="ignore"):
        scores_max = scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores - scores_max)
        attn = attn / attn.sum(axis=-1, keepdims=True)
    return np.einsum("bhij,bjhd->bihd", attn, v)


def time_fn(fn, n_iter: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


def main() -> None:
    parser = argparse.ArgumentParser(description="scree CPU throughput vs padded")
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--mean-len", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    lengths = realistic_lengths(args.num_seqs, args.mean_len, args.sigma)
    max_len = max(lengths)
    total = sum(lengths)
    n_heads, head_dim = args.n_heads, args.head_dim

    print(f"workload: {args.num_seqs} seqs, mean_len={args.mean_len}, max_len={max_len}")
    print(f"          n_heads={n_heads}, head_dim={head_dim}")
    print(f"          {total} real tokens vs {args.num_seqs * max_len} padded "
          f"({(args.num_seqs * max_len - total) / total:.1f}× padding overhead)")
    print()

    # scree input (packed)
    seqs_q = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_k = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    seqs_v = [rng.standard_normal((n, n_heads, head_dim)).astype(np.float32) for n in lengths]
    q_arr = scree.pack(seqs_q)
    k_arr = scree.pack(seqs_k)
    v_arr = scree.pack(seqs_v)

    # Padded input
    q_pad = np.zeros((args.num_seqs, max_len, n_heads, head_dim), dtype=np.float32)
    k_pad = np.zeros_like(q_pad)
    v_pad = np.zeros_like(q_pad)
    mask = np.zeros((args.num_seqs, max_len), dtype=bool)
    for i, n in enumerate(lengths):
        q_pad[i, :n] = seqs_q[i]
        k_pad[i, :n] = seqs_k[i]
        v_pad[i, :n] = seqs_v[i]
        mask[i, :n] = True

    # ----- varlen_attention vs padded_attention -----
    print("varlen_attention (causal=False) vs padded baseline:")
    scree_ms = time_fn(lambda: varlen_attention(q_arr, k_arr, v_arr), args.iters)
    pad_ms = time_fn(lambda: padded_attention(q_pad, k_pad, v_pad, mask), args.iters)
    print(f"  scree (packed):  {scree_ms:7.2f} ms")
    print(f"  padded baseline: {pad_ms:7.2f} ms")
    print(f"  scree speedup:   {pad_ms / scree_ms:5.2f}x")
    print()

    # ----- varlen_rmsnorm vs naive padded rmsnorm -----
    print("varlen_rmsnorm vs padded baseline:")
    x_arr = scree.pack([rng.standard_normal((n, n_heads * head_dim)).astype(np.float32) for n in lengths])
    x_pad = np.zeros((args.num_seqs, max_len, n_heads * head_dim), dtype=np.float32)
    for i, n in enumerate(lengths):
        x_pad[i, :n] = rng.standard_normal((n, n_heads * head_dim)).astype(np.float32)

    def padded_rmsnorm() -> np.ndarray:
        rms = np.sqrt((x_pad * x_pad).mean(axis=-1, keepdims=True) + 1e-6)
        return x_pad / rms

    scree_ms = time_fn(lambda: varlen_rmsnorm(x_arr), args.iters)
    pad_ms = time_fn(padded_rmsnorm, args.iters)
    print(f"  scree (packed):  {scree_ms:7.2f} ms")
    print(f"  padded baseline: {pad_ms:7.2f} ms")
    print(f"  scree speedup:   {pad_ms / scree_ms:5.2f}x")


if __name__ == "__main__":
    main()
