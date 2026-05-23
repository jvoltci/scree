# FAQ

Questions that come up after a few minutes with scree.

## What is scree, in one sentence?

A typed primitive for variable-length sequence data — like NumPy gave us `ndarray`, scree gives us `Array` — with reference kernels for varlen attention/layernorm/softmax and an autotuned Triton implementation.

## Why is it called scree?

A scree is the irregular pile of rock fragments accumulated on a mountain slope. Variable-length sequences pack against each other the same way: irregular shapes, fitted by their irregularity, not despite it.

## Do I have to use the Triton kernels?

No. `scree.Array` is a plain data structure with NumPy/PyTorch/MLX backends. The reference kernels (`scree.kernels.reference.*`) work everywhere. The Triton kernels (`scree.kernels.triton.*`) are an optional fast path for CUDA users. The reference impls are the contract; the Triton impls are the speed.

## How does scree compare to `torch.nested`?

`torch.nested` is PyTorch's jagged layout — similar concept, PyTorch-only.

| Feature | `torch.nested` | `scree` |
| --- | --- | --- |
| Frameworks | PyTorch only | NumPy + PyTorch + MLX (JAX next) |
| Status | beta since 2021, partial coverage | pre-alpha but with all 5 core ops |
| Apple Silicon | works but suboptimal | native via MLX backend |
| Cross-framework | not a goal | the main goal |
| Bridges | imports only torch | bridges to torch.nested, HF padded, FA cu_seqlens, DLPack |

`scree.bridges.to_torch_nested` and `from_torch_nested` round-trip between the two. Use whichever fits your stack.

## How does scree compare to FlashAttention's `cu_seqlens`?

`cu_seqlens` *is* scree's `offsets`. They're the same int32 buffer with the same `[0, L0, L0+L1, ...]` layout. `scree.from_cu_seqlens(values, cu_seqlens)` is zero-copy — the resulting `Array` shares memory with the inputs.

Once you have a `scree.Array`, you can pass it around your codebase as a typed value, then unpack at the FlashAttention call. scree doesn't replace FlashAttention; it provides the type that FlashAttention's API would have had if it shipped one.

## Can I use it for training?

Today: yes for forward passes. The reference kernels and the Triton forward kernel are autograd-compatible (gradients flow through `values`). The backward pass for `varlen_attention_triton` is the v0.1 → v0.2 work.

For now, use the reference kernels for training (slow but correct), or use FlashAttention varlen for the attention op and scree for everything else.

## Can I use it for inference?

Yes, with caveats. The forward kernel is fast enough for inference (within 1.21× of FA-2 on H100). What's missing for production inference:

- **KV cache append** — concatenating new tokens to an existing KV cache. Currently you'd have to do it in PyTorch ops outside the kernel.
- **Paged KV cache** — vLLM-style block-paged storage. Not yet supported.
- **Continuous batching adapter** — integration with vLLM/SGLang as a batch format. Planned for v0.2.

For static-batch inference (e.g., evals), scree works today.

## What about JAX?

JAX backend is the most-requested feature for v0.1. The main work is adapting `pack`/`unpack`/`to_padded`/`from_padded` to JAX's immutable arrays (similar to the MLX adaptation done in commit `4876a21`). Tracked in [architecture.md](architecture.md) "Adding a new backend."

## What about MLX (Apple Silicon)?

Supported in v0.0 — see [getting-started.md](getting-started.md). MLX is a first-class backend; all reference kernels work, and MLX matmul uses tensor-core mixed precision (similar to NVIDIA's TF32), so cross-backend agreement with NumPy is to ~5e-3.

No Apple-Silicon-specific kernel yet (the Triton-equivalent for Metal is MLX's own `mlx.fast.*` modules; v0.2 work).

## How big is the v0.0 footprint?

The full package is about 1500 lines of Python + Triton across:

- `src/scree/_core.py` — ~200 lines (Array + 5 ops)
- `src/scree/bridges/` — ~150 lines (4 bridges)
- `src/scree/kernels/reference/` — ~250 lines (3 reference kernels × 3 backends each)
- `src/scree/kernels/triton/varlen_attention.py` — ~250 lines (the GPU kernel)
- `tests/` — ~500 lines (31 tests)

Plus benchmarks and docs.

## Does it support [my framework / hardware]?

| Framework | Status |
| --- | --- |
| NumPy | ✅ |
| PyTorch | ✅ |
| MLX (Apple Silicon) | ✅ |
| JAX | planned for v0.1 |
| TensorFlow | not planned |
| CuPy | possible via Array API |

| Hardware | Triton kernel? |
| --- | --- |
| NVIDIA Ampere / Hopper / Blackwell | ✅ |
| AMD MI200/MI300 | possible (Triton has experimental ROCm support) |
| Apple Silicon GPU | via MLX, not via Triton |
| Intel GPU | not planned |
| TPU | via JAX backend (planned) |

## Can I install scree alongside torch.nested?

Yes. They don't conflict. `scree.bridges.to_torch_nested` and `from_torch_nested` exist precisely so you can use both.

## Will scree publish on conda-forge?

Not for v0.0. Once v0.1 ships and the API is stable, yes.

## Is scree fast on CPU?

The reference kernels are explicitly Python-loop-based. They're correct, not fast. For CPU-only workloads at scale, the right approach is to use `to_torch` or `to_padded` and run torch's CPU kernels on the dense form.

A CPU-optimized scree path (Halide-style or SIMD intrinsics) is not planned — most users running on CPU don't need varlen kernels at the speeds scree targets.

## Why no PyPI release yet?

scree is at v0.0.x — the API may break between commits. v0.1 will be the first PyPI-published release. Install from source for now:

```bash
git clone https://github.com/jvoltci/scree
cd scree
pip install -e ".[torch]"
```

## Where do I file a bug?

GitHub Issues: <https://github.com/jvoltci/scree/issues>

Please include:
- Your Python version, OS, scree version
- Backend (NumPy / PyTorch / MLX) and version of that backend
- For Triton bugs: GPU model, CUDA driver version, Triton version
- A minimal reproduction (~10 lines)

## How do I propose a new feature?

Open a GitHub Discussion first (1-2 paragraphs is enough). The maintainers will respond with whether it fits scree's scope and what the API should look like. Then PRs follow from the discussion.

See [../CONTRIBUTING.md](../CONTRIBUTING.md).

## What's the long-term plan?

See [../docs/architecture.md](architecture.md) → "Versioning" for the v0.0 / v0.1 / v1.0 milestone gates. In short:

- **v0.0** (now) — primitive + bridges + reference kernels + first Triton kernel
- **v0.1** — JAX backend + Triton backward + autotuned Triton at FA-2 parity + PyPI release
- **v0.2** — vLLM/SGLang adapter, GQA, KV-cache append, multimodal segment metadata
- **v1.0** — API stability commitment, conda-forge, broad ecosystem integration
