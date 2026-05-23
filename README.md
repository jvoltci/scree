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
out = varlen_attention(arr, arr, arr, causal=True)
```

## Why

Variable-length sequence data is everywhere in modern ML — transformer
training, inference batching, multimodal interleaving, MoE routing — yet
every team carries their own incompatible representation:

- `torch.nested` (PyTorch only, in beta since 2021)
- TF `RaggedTensor` (TensorFlow only)
- FlashAttention `cu_seqlens` (a convention, not a typed primitive)
- vLLM / SGLang packed batches (internal data structures)
- HuggingFace `attention_mask` (pads, then masks — wasting memory and FLOPs)

`scree` ships one primitive — a packed `values + offsets + ragged_dim`
array — that bridges across frameworks and ships with reference varlen
kernels for attention, layernorm, softmax, and scatter/gather.

## What you get

**Memory savings vs HF padded** on realistic LLM length distributions
(log-normal):

| Workload | Mean savings | Min – Max |
| --- | --- | --- |
| Training-style (batch 64, mean_len 256, σ=0.6) | **71%** | 63% – 84% |
| Inference-style (batch 32, mean_len 1024, σ=1.2) | **85%** | 75% – 94% |

Reproduce: `python benchmarks/bench_memory.py`

**GPU kernel parity** with FlashAttention-2 — first-attempt unautotuned
Triton kernel on H100, 16 seqs × log-normal lengths, 12160 total tokens,
16 heads × head_dim 64, fp16, causal:

| Kernel | Time | Ratio |
| --- | --- | --- |
| FlashAttention-2 varlen | 0.166 ms | 1.00x |
| **scree-Triton varlen** | **0.201 ms** | **1.21x** |

Correctness: max abs diff 4.88e-4 vs FlashAttention-2 (PASS).
Reproduce: `modal run benchmarks/modal_bench.py` (~$0.20 of Modal credit).

**Zero-copy bridges** to the things you already use:

```python
import scree.bridges as bridges

arr = scree.from_cu_seqlens(values, cu_seqlens)         # FlashAttention
arr = bridges.from_hf_padded(hidden_states, attn_mask)  # HuggingFace
arr = bridges.from_torch_nested(nt)                     # torch.nested

bridges.to_torch_nested(arr)   # → torch.NestedTensor
bridges.to_hf_padded(arr)      # → (hidden_states, attention_mask)
bridges.to_torch(arr)          # numpy values → torch tensors via DLPack
```

**One primitive, every framework** — values and offsets can be NumPy,
PyTorch, or MLX (Apple Silicon, via Metal) today; JAX is next.

## The name

A scree is the irregular pile of rock fragments accumulated on a mountain
slope. Variable-length sequences pack against each other the same way:
irregular shapes, fitted by their irregularity, not despite it.

## Status

v0.0.1, pre-alpha. The reference (slow but correct) Python kernels are
present. Triton kernels at FlashAttention-varlen parity ship in v0.1.

| Component | Status |
| --- | --- |
| `scree.Array` type + invariants | ✅ |
| `pack` / `unpack` / `to_padded` / `from_padded` | ✅ |
| Reference varlen attention / layernorm / softmax | ✅ |
| Bridges: torch.nested, HF padded, FA cu_seqlens, DLPack | ✅ |
| NumPy + PyTorch + MLX backends | ✅ |
| Triton kernels at FA-varlen parity | 🟡 next |
| JAX backend | 🟡 next |

## Install

```bash
pip install scree              # numpy backend
pip install "scree[torch]"     # + PyTorch backend
pip install "scree[mlx]"       # + MLX backend (Apple Silicon, Metal)
```

## Examples

- [`examples/01_quickstart.py`](examples/01_quickstart.py) — pack/unpack + varlen attention
- [`examples/02_no_pad_transformer.py`](examples/02_no_pad_transformer.py) — full transformer block, zero padding

## License

Apache-2.0
