# scree: a cross-framework primitive for variable-length sequence data

*Draft for v0.1.0 launch on scree.dev. Not yet published. Owner: maintainer.*

---

> One `values + offsets + ragged_dim` array. Four backends. Six bridges. Triton
> attention at parity with FlashAttention-2. 71–85% memory savings vs HF padded.
> Apache-2.0. v0.1 today.

## The thing the field has been working around

Variable-length sequence data is everywhere in modern ML — transformer training,
inference batching, multimodal interleaving, MoE routing, RAG, graph neural
networks. Every team uses it. And every team writes their own incompatible
plumbing for it:

- `torch.nested` (Meta) is PyTorch-only and has been "almost ready" since 2021.
- `RaggedTensor` (Google) is TensorFlow-only; the TF ecosystem is fading.
- FlashAttention's `cu_seqlens` is a *convention*, not a typed primitive.
- vLLM and SGLang each carry internal packed-batch structures, not exposed.
- HuggingFace Transformers pads, then masks — wasting memory and FLOPs on the padding.
- JAX has no ragged story at all. Issue [#17863](https://github.com/jax-ml/jax/issues/17863) has been open since 2023.

The math is one concept (a flat buffer of values + offsets pointing at row
boundaries) but the implementations refuse to converge. A team building a
no-pad transformer can't share a batch between trainer and inference engine.
A multimodal builder reinvents the interleaving logic every model. Anyone
trying to migrate kernels between frameworks pays a bookkeeping tax on every
edge.

This is the gap scree fills.

## What scree is

One data type:

```python
import scree
import numpy as np

seqs = [np.random.randn(n, 8).astype(np.float32) for n in [4, 2, 7]]
arr = scree.pack(seqs)
# arr.values  : shape (13, 8) — one flat buffer
# arr.offsets : [0, 4, 6, 13] — row boundaries
```

Five operations on it (pack, unpack, to_padded, from_padded, from_cu_seqlens).
Six bridges to the existing world (torch.nested ↔ scree, HF padded ↔ scree,
FlashAttention cu_seqlens ↔ scree, plus DLPack cross-framework). Four backends
that all pass the same correctness suite: **NumPy, PyTorch, MLX (Apple Silicon
via Metal), JAX**.

And four reference kernels — `varlen_attention` (causal and non-causal),
`varlen_layernorm`, `varlen_rmsnorm`, `varlen_softmax` — plus a FlashAttention-2
style Triton kernel that hits **1.21× of FA-2 varlen** on H100 on its first
attempt, unautotuned, with PASS correctness.

That's the whole library. ~1500 lines of Python + Triton.

## Three numbers

These are all reproducible. The benchmark scripts ship in the repo.

**Memory:** scree's packed representation is **71–85% smaller** than the
equivalent HuggingFace `(hidden_states, attention_mask)` representation on
realistic log-normal LLM length distributions.

| Workload | Mean savings |
| --- | --- |
| Training-style (batch 64, mean_len 256, σ=0.6) | **71%** |
| Inference-style (batch 32, mean_len 1024, σ=1.2) | **85%** |

Reproduce: `python benchmarks/bench_memory.py`.

**CPU throughput:** scree's reference varlen_attention is **6.6× faster** than
a naive padded baseline (no mask-skip optimization, which is what HF Transformers
defaults to) on a realistic 16-sequence batch.

Reproduce: `python benchmarks/bench_throughput.py`.

**GPU parity:** scree's Triton `varlen_attention_triton` is **1.21× of
FlashAttention-2 varlen** on H100, fp16, causal, 12k total tokens.

Reproduce: `modal run benchmarks/modal_bench.py` (~$0.20 of Modal credit).

## Why it composes

The `values + offsets` layout is the same layout FlashAttention's varlen API
takes (`cu_seqlens` is just our `offsets`). It's the same layout vLLM and SGLang
use internally for continuous batching. It's the layout `torch.nested`
jagged-layout exposes through `.values()` and `.offsets()`. scree picks the
representation everyone has already converged on for performance reasons and
makes it a first-class typed value.

This means migration is one line. Wherever your code today has a
`(hidden_states, attention_mask)` pair, swap in:

```python
arr = scree.bridges.from_hf_padded(hidden_states, attention_mask)
```

Wherever you're packing with FlashAttention:

```python
arr = scree.from_cu_seqlens(values, cu_seqlens)   # zero-copy
```

And when you need to hand the result back to existing code:

```python
hidden_states, attention_mask = scree.bridges.to_hf_padded(arr)
```

The round-trip is bit-exact for layernorm and softmax (no recomputation, no
fp drift). For matmul-heavy ops it agrees with the HF-native path to fp
tolerance.

## Why it's not just a wrapper

A primitive without kernels is a Pydantic model. We took kernels seriously:

- The reference kernels are written three times (NumPy, PyTorch, MLX) and run
  through the same numerical agreement tests. JAX joined as the fourth backend
  in v0.1 with all four reference kernels passing.
- The Triton GPU kernel uses the FlashAttention-2 style online-softmax
  recurrence. It's not novel — it's the canonical algorithm. What's novel
  is shipping it under a typed primitive that lets you mix backends.
- Autograd flows through `scree.Array` transparently. A 30-step training loop
  on a tiny transformer drops loss 80× — proof in
  `examples/03_train_step.py`.

## What scree is NOT

scree is one primitive, not a framework. We deliberately don't:

- Ship a training loop (you bring your own).
- Define an attention API for a specific model class.
- Care which inference engine you use.
- Provide a model zoo.

The library's surface is small on purpose. The bet is that other libraries —
HF Transformers integrations, no-pad model classes, varlen-native dataloaders —
will be built on top, by people closer to those use cases than we are.

## What's in v0.1

Pre-flight: 56 tests passing across 4 backends including 11 property-based
tests via Hypothesis covering pack/unpack roundtrip, offset invariants,
softmax row-sum, layernorm zero-mean unit-var, rmsnorm unit-rms, and attention
shape preservation.

CI on every push across Python 3.10/3.11/3.12 (Ubuntu) + 3.11 (macOS) running
all tests + every example + the memory benchmark.

Four working examples: quickstart, no-pad transformer block, training step
with autograd, HuggingFace compatibility recipe.

Full documentation tree: getting-started, concepts, API reference, bridges,
kernels, architecture, benchmarks, FAQ.

Honest deferrals to v0.2:

- Triton autotune (hit a known Triton 3.0 Hopper compiler bug; revisit when
  Triton 3.1+ is available on Modal).
- Triton backward pass for varlen_attention.
- vLLM / SGLang integration as a canonical batch format.
- Multimodal segment metadata.

## How to use it today

```bash
pip install "scree[torch]"
```

Or `pip install scree` for NumPy-only, `pip install "scree[mlx]"` for Apple
Silicon, `pip install "scree[jax]"` for JAX.

Quickstart at [github.com/scree-dev/scree/blob/main/examples/01_quickstart.py](https://github.com/scree-dev/scree).
Migration cookbook at [docs/bridges.md](bridges.md). API reference at
[docs/api.md](api.md).

## How to help

- Tell us if it doesn't work for your workload. Issues:
  [github.com/scree-dev/scree/issues](https://github.com/scree-dev/scree/issues).
- The biggest open question: vLLM and SGLang would both benefit from accepting
  `scree.Array` as a canonical batch format. We have draft integration PRs
  but need their teams' input. If you're on either team, let's talk.
- We're particularly interested in feedback from anyone running on JAX, MLX,
  or AMD ROCm — the three backends we have the least testing on.

scree is Apache-2.0. Contributions follow the
[CONTRIBUTING](https://github.com/scree-dev/scree/blob/main/CONTRIBUTING.md)
flow.

## Acknowledgments

The Triton `varlen_attention` forward kernel is structurally a port of the
patterns Tri Dao and collaborators established with FlashAttention. The
type design is influenced by every team that built a packed representation
before us — vLLM, SGLang, FlashAttention, `torch.nested`,
`tf.RaggedTensor`. We owe each of them.

— maintainer
v0.1.0, 2026
