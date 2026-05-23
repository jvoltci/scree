# Getting started

## Install

```bash
pip install scree                    # NumPy backend
pip install "scree[torch]"           # + PyTorch (recommended)
pip install "scree[mlx]"             # + MLX (Apple Silicon, Metal)
pip install "scree[torch,mlx]"       # + both
```

scree supports Python 3.10 and newer. NumPy is the only required dependency; PyTorch and MLX are optional and detected at runtime.

## Your first scree program

```python
import numpy as np
import scree

# Three sequences of different lengths.
seqs = [np.random.randn(n, 8).astype(np.float32) for n in [4, 2, 7]]

# Pack them into one scree.Array — no padding.
arr = scree.pack(seqs)

print(arr)
# scree.Array(batch_size=3, total_length=13, feature_shape=(8,), dtype=float32)

print(arr.values.shape)     # (13, 8) — packed buffer, 4+2+7 = 13 rows
print(arr.offsets.tolist()) # [0, 4, 6, 13] — row boundaries

# Unpack back to a list of arrays.
for i, row in enumerate(scree.unpack(arr)):
    print(f"row {i}: shape={row.shape}")
# row 0: shape=(4, 8)
# row 1: shape=(2, 8)
# row 2: shape=(7, 8)
```

That's the whole library: one type, a flat values buffer plus offsets marking row boundaries.

## Three patterns you'll use constantly

### 1. Pack a list of variable-length arrays

```python
seqs = [np.random.randn(n, 4).astype(np.float32) for n in [3, 5, 2]]
arr = scree.pack(seqs)
```

`scree.pack` works for NumPy, PyTorch tensors, or MLX arrays. The backend of the first array determines the backend of the result.

```python
import torch
seqs_t = [torch.randn(n, 4) for n in [3, 5, 2]]
arr_t = scree.pack(seqs_t)
# arr_t.values is a torch.Tensor; arr_t.offsets is a torch.Tensor

import mlx.core as mx
seqs_m = [mx.random.normal((n, 4)) for n in [3, 5, 2]]
arr_m = scree.pack(seqs_m)
# arr_m.values is an mlx.core.array
```

### 2. Bridge to/from your existing pipeline

```python
import scree.bridges as bridges

# Coming from HuggingFace Transformers
arr = bridges.from_hf_padded(hidden_states, attention_mask)

# Going to HuggingFace
hidden_states, attention_mask = bridges.to_hf_padded(arr)

# Coming from FlashAttention — cu_seqlens IS the offsets format, zero-copy
arr = scree.from_cu_seqlens(values, cu_seqlens)

# Going to PyTorch's jagged NestedTensor
nt = bridges.to_torch_nested(arr)

# Cross-framework via DLPack
arr_torch = bridges.to_torch(arr_numpy)
arr_numpy = bridges.to_numpy(arr_torch)
```

See [bridges.md](bridges.md) for the full migration cookbook.

### 3. Run a varlen kernel on a scree.Array

```python
from scree.kernels.reference import varlen_attention, varlen_layernorm, varlen_softmax

# q, k, v are scree.Arrays with the same offsets (same batch structure).
# Each sequence attends only to itself — no cross-sequence attention.
out = varlen_attention(q, k, v, causal=True)

# Layernorm over the last (feature) dim. Per-token; no cross-row interaction.
y = varlen_layernorm(arr, eps=1e-5)

# Softmax along the ragged dim — per-sequence softmax, not global.
p = varlen_softmax(scores)
```

The reference kernels in `scree.kernels.reference` are slow Python implementations. They exist as the ground truth used in CI tests of the optimized Triton kernels. For GPU speed, use `scree.kernels.triton.varlen_attention_triton` (CUDA-only).

## Five-line memory-savings demo

```python
import numpy as np, scree
from scree.bridges import to_hf_padded
seqs = [np.random.randn(n, 4096).astype(np.float32) for n in [128, 256, 768, 64, 1024]]
arr = scree.pack(seqs)
padded, mask = to_hf_padded(arr)
print(f"scree:  {(arr.values.nbytes + arr.offsets.nbytes) / 1e6:.1f} MB")
print(f"padded: {(padded.nbytes + mask.nbytes) / 1e6:.1f} MB")
# scree:  36.9 MB
# padded: 84.0 MB   (55% wasted on padding tokens)
```

## What to read next

- If you're moving an existing project to scree: [bridges.md](bridges.md)
- If you want to understand the design: [concepts.md](concepts.md)
- If you're looking up a specific function: [api.md](api.md)
- If you want to contribute: [architecture.md](architecture.md) + [CONTRIBUTING](../CONTRIBUTING.md)
