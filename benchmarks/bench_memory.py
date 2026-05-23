"""Memory benchmark: scree.Array vs HuggingFace padded representation.

Measures the in-memory footprint of scree's packed ``values + offsets``
against HuggingFace's ``(hidden_states, attention_mask)`` on workloads
with realistic LLM sequence-length distributions.

Runs on CPU — no GPU required.

Usage
-----
    python benchmarks/bench_memory.py
    python benchmarks/bench_memory.py --num-seqs 128 --feature-dim 4096 --mean-len 512
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import numpy as np

import scree
from scree.bridges import to_hf_padded


@dataclass
class Stats:
    scree_bytes: int
    padded_bytes: int
    valid_tokens: int
    padded_tokens: int

    @property
    def savings_pct(self) -> float:
        return (1.0 - self.scree_bytes / self.padded_bytes) * 100.0

    @property
    def waste_ratio(self) -> float:
        return self.padded_tokens / self.valid_tokens


def measure(lengths: list[int], feature_dim: int, dtype: np.dtype = np.float32) -> Stats:
    """Compute exact byte counts for both representations on identical data."""
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, feature_dim)).astype(dtype) for n in lengths]
    arr = scree.pack(seqs)
    scree_bytes = arr.values.nbytes + arr.offsets.nbytes

    padded, mask = to_hf_padded(arr)
    padded_bytes = padded.nbytes + mask.nbytes

    return Stats(
        scree_bytes=scree_bytes,
        padded_bytes=padded_bytes,
        valid_tokens=int(arr.total_length),
        padded_tokens=len(lengths) * max(lengths),
    )


def realistic_lengths(num_seqs: int, mean_len: int, sigma: float = 0.6, seed: int = 0) -> list[int]:
    """Sample lengths from a log-normal — the empirical shape of real LLM batches.

    Most sequences are short; a few are long. This is why HF padding wastes
    so much memory: the worst-case sequence sets the budget for everyone.
    """
    rng = np.random.default_rng(seed)
    lengths = rng.lognormal(mean=np.log(mean_len), sigma=sigma, size=num_seqs)
    return [max(1, int(length)) for length in lengths]


def main() -> None:
    parser = argparse.ArgumentParser(description="scree memory benchmark vs HF padded")
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--mean-len", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.6, help="log-normal sigma; higher = longer tail")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    savings_list = []
    waste_list = []
    for trial in range(args.trials):
        lengths = realistic_lengths(args.num_seqs, args.mean_len, sigma=args.sigma, seed=trial)
        stats = measure(lengths, args.feature_dim)
        savings_list.append(stats.savings_pct)
        waste_list.append(stats.waste_ratio)
        print(
            f"trial {trial}: {stats.valid_tokens:>7} valid / {stats.padded_tokens:>7} padded "
            f"tokens   waste={stats.waste_ratio:4.2f}x   savings={stats.savings_pct:5.1f}%"
        )

    print()
    print(f"batch size:        {args.num_seqs}")
    print(f"feature dim:       {args.feature_dim}")
    print(f"mean seq length:   {args.mean_len}  (log-normal sigma={args.sigma})")
    print()
    print(
        f"memory savings:    {statistics.mean(savings_list):5.1f}% mean   "
        f"{statistics.median(savings_list):5.1f}% median   "
        f"{min(savings_list):5.1f}% min   {max(savings_list):5.1f}% max"
    )


if __name__ == "__main__":
    main()
