"""Multimodal interleaved data in one scree.Array.

Demonstrates the use case that's painful in HF Transformers but natural
in scree: each sequence is a mixed stream of text tokens and image
patches, with different sequences having different total lengths AND
different splits between text and image. scree packs them into one
flat buffer; attention runs once across the whole packed batch.

Real workloads this maps to:
  - LLaVA-style vision-language models (image patches + text tokens)
  - audio-text models (audio frames + text tokens)
  - multi-image VQA (multiple images of varying patch counts + text)
  - tool-augmented LLMs (text + structured tool-output blobs)

Run:
    python examples/05_multimodal_interleaved.py
"""

from __future__ import annotations

import numpy as np

import scree
from scree.kernels.reference import varlen_attention


def make_sequence(rng, text_len: int, n_images: int, patches_per_image: int, dim: int):
    """Build a sequence that interleaves text tokens and image patches.

    Layout for each sequence:
      [text_block_1, image_1_patches, text_block_2, image_2_patches, ..., text_block_last]
    """
    blocks = []
    text_remaining = text_len
    for _ in range(n_images):
        # Split text roughly evenly between images
        text_chunk = text_remaining // (n_images + 1)
        if text_chunk > 0:
            blocks.append(rng.standard_normal((text_chunk, dim)).astype(np.float32))
            text_remaining -= text_chunk
        blocks.append(rng.standard_normal((patches_per_image, dim)).astype(np.float32))
    if text_remaining > 0:
        blocks.append(rng.standard_normal((text_remaining, dim)).astype(np.float32))
    return np.concatenate(blocks, axis=0)


def main() -> None:
    rng = np.random.default_rng(0)
    dim = 32
    n_heads = 2
    head_dim = dim // n_heads

    # Four heterogeneous sequences with different multimodal compositions:
    #   sequence 0: short text + 1 image
    #   sequence 1: long text + no images (pure text)
    #   sequence 2: medium text + 3 images
    #   sequence 3: very long text + 2 images
    seq_specs = [
        ("short text + 1 image",  20, 1, 16),
        ("long text only",        60, 0, 16),
        ("medium text + 3 images", 30, 3, 8),
        ("long text + 2 images",  80, 2, 12),
    ]
    seqs = [make_sequence(rng, text_len=t, n_images=n, patches_per_image=p, dim=dim)
            for _, t, n, p in seq_specs]

    print("=== input ===")
    for (label, t, n, p), seq in zip(seq_specs, seqs):
        total = seq.shape[0]
        print(f"  {label:30s}  text={t:3d} images={n:1d} ({p:2d} patches each)  "
              f"total tokens = {total}")
    print()

    # Pack into a single scree.Array — no special multimodal API needed.
    # The interleaving is just bytes in the flat buffer; offsets demarcate sequences.
    arr = scree.pack(seqs)
    print(f"=== packed ===")
    print(f"  {arr}")
    print(f"  offsets:        {arr.offsets.tolist()}")
    print(f"  lengths:        {arr.lengths.tolist()}")
    print(f"  total tokens:   {arr.total_length}")
    print(f"  padded would be: {arr.batch_size * max(arr.lengths.tolist())}")
    print(f"  memory saved:   {(1 - arr.total_length / (arr.batch_size * max(arr.lengths.tolist()))) * 100:.0f}%")
    print()

    # Reshape values to (total, n_heads, head_dim) for attention.
    qkv_shaped = arr.values.reshape(-1, n_heads, head_dim)
    q = scree.Array(values=qkv_shaped, offsets=arr.offsets)
    k = scree.Array(values=qkv_shaped, offsets=arr.offsets)
    v = scree.Array(values=qkv_shaped, offsets=arr.offsets)

    # Causal varlen attention. Each sequence attends ONLY to itself —
    # text in sequence 0 cannot see image patches in sequence 3, even
    # though they share the same packed buffer.
    out = varlen_attention(q, k, v, causal=True)
    print(f"=== output ===")
    print(f"  {out}")
    for i, row in enumerate(scree.unpack(out)):
        label = seq_specs[i][0]
        print(f"  seq {i} ({label}): output shape {row.shape}")

    print()
    print("Key property: image patches and text tokens are mixed in one buffer,")
    print("but cross-sequence attention is structurally impossible — offsets")
    print("partition the buffer into independent attention contexts.")


if __name__ == "__main__":
    main()
