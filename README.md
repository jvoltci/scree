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

**CPU throughput** vs a naive padded attention baseline on a real
batch (16 seqs × log-normal lengths, 1980 real / 4464 padded tokens,
4 heads × head_dim 32, fp32, no mask optimization):

| Operation | scree | padded baseline | Speedup |
| --- | --- | --- | --- |
| varlen_attention | 34.7 ms | 228.3 ms | **6.6×** |
| varlen_rmsnorm | 0.13 ms | 0.28 ms | 2.1× |

Reproduce: `python benchmarks/bench_throughput.py`

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
PyTorch, MLX (Apple Silicon, via Metal), or JAX. All four backends pass
the same correctness suite end-to-end.

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
| NumPy + PyTorch + MLX + JAX backends | ✅ |
| Triton kernels at FA-varlen parity | 🟡 next |
| Triton autotune (Triton 3.1+) | 🟡 next |

## Install

```bash
pip install scree              # numpy backend
pip install "scree[torch]"     # + PyTorch backend
pip install "scree[mlx]"       # + MLX backend (Apple Silicon, Metal)
pip install "scree[jax]"       # + JAX backend
```

## Examples

- [`examples/01_quickstart.py`](examples/01_quickstart.py) — pack/unpack + varlen attention
- [`examples/02_no_pad_transformer.py`](examples/02_no_pad_transformer.py) — full transformer block, zero padding

## Documentation

- [**Getting started**](docs/getting-started.md) — install, first program, common patterns
- [**Concepts**](docs/concepts.md) — the mental model behind `values + offsets + ragged_dim`
- [**API reference**](docs/api.md) — every public function and class
- [**Bridges & migration**](docs/bridges.md) — moving from `torch.nested`, HuggingFace, FlashAttention
- [**Kernels**](docs/kernels.md) — reference and Triton kernel design
- [**Architecture**](docs/architecture.md) — internal layout for contributors
- [**Benchmarks**](docs/benchmarks.md) — methodology and reproduction
- [**FAQ**](docs/faq.md)

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow. Open a GitHub Discussion for anything beyond a small fix.

## License

Apache-2.0
