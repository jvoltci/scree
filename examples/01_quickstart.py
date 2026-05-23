"""scree quickstart — variable-length sequences without padding.

Demonstrates pack/unpack and the memory savings vs padded representations.
"""

import numpy as np

import scree
from scree.kernels.reference import varlen_attention


def main() -> None:
    # Three sequences of different lengths. Real workloads have batches like
    # this: tokenized prompts, MoE routed tokens, audio frames, image patches.
    lengths = [4, 2, 7]
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 1, 8)).astype(np.float32) for n in lengths]

    # Pack into one scree.Array — no padding.
    arr = scree.pack(seqs)
    padded_total = arr.batch_size * max(lengths)

    print(arr)
    print(f"  total tokens: {arr.total_length}")
    print(f"  padded would be: {padded_total}")
    print(f"  memory saved: {(1 - arr.total_length / padded_total) * 100:.0f}%")

    # Run varlen attention. Each sequence attends only to itself.
    out = varlen_attention(arr, arr, arr, causal=True)
    print(f"\noutput: {out}")

    # Unpack to a list of arrays (views into the same buffer).
    for i, row in enumerate(scree.unpack(out)):
        print(f"  row {i}: shape={row.shape}")

    # Convert to padded form when downstream code requires it.
    padded, mask = scree.to_padded(arr)
    print(f"\npadded shape: {padded.shape}, mask shape: {mask.shape}")


if __name__ == "__main__":
    main()
