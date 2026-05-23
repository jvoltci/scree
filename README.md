# scree

A cross-framework ragged tensor primitive for variable-length sequence data.

```python
import scree
import numpy as np

# Three sequences of different lengths.
seqs = [np.random.randn(n, 8).astype(np.float32) for n in [4, 2, 7]]

# Pack them into one scree.Array — no padding.
arr = scree.pack(seqs)
# arr.values: shape (13, 8), arr.offsets: [0, 4, 6, 13]

# Run varlen attention. Each sequence attends only to itself.
from scree.kernels.reference import varlen_attention
out = varlen_attention(arr, arr, arr)
```

## The problem

Variable-length sequence data is everywhere in modern ML — transformer
training, inference batching, multimodal interleaving, MoE routing — yet
every team carries their own incompatible representation:

- `torch.nested` (PyTorch only, in beta since 2021)
- TF `RaggedTensor` (TensorFlow only)
- FlashAttention `cu_seqlens` (a convention, not a typed primitive)
- vLLM / SGLang packed batches (internal data structures)
- HuggingFace `attention_mask` (pads, then masks)

`scree` ships one primitive — a packed `values + offsets + ragged_dim`
array — that bridges across frameworks and ships with reference varlen
kernels for attention, layernorm, softmax, and scatter/gather.

## The name

A scree is the irregular pile of rock fragments accumulated on a mountain
slope. Variable-length sequences pack against each other the same way:
irregular shapes, fitted by their irregularity, not despite it.

## Status

v0.0.1, pre-alpha. The reference (slow but correct) implementations are
present; Triton kernels at FlashAttention-varlen parity ship in the next
release.

## Install

```bash
pip install scree              # numpy backend
pip install "scree[torch]"     # + PyTorch backend
```

## License

Apache-2.0
