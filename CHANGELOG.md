# Changelog

All notable changes to scree are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Comprehensive documentation under `docs/` covering concepts, API reference, bridges, kernels, architecture, benchmarks, and FAQ
- `CONTRIBUTING.md` and `CHANGELOG.md`
- `varlen_rmsnorm` reference kernel — the norm used by LLaMA / Mistral / Mixtral / DeepSeek / Qwen and most modern open transformers (replaces LayerNorm in nearly every post-2023 architecture)
- `benchmarks/bench_throughput.py` — CPU throughput benchmark. **6.6× faster than padded baseline** on `varlen_attention` for a realistic 16-sequence batch.
- **JAX backend.** All four reference kernels (varlen_attention, varlen_layernorm, varlen_rmsnorm, varlen_softmax) now run on JAX arrays. Cross-backend numerical agreement vs NumPy reference (atol 1e-4 for matmul-heavy ops, 1e-5 otherwise). Closes the v0.1 "cross-framework" claim: NumPy + PyTorch + MLX + JAX all pass the same correctness suite. Total test count 31 → 40.

### Investigated (deferred)
- Attempted `@triton.autotune` over a 24-config grid for `varlen_attention`. Hit a known Triton 3.0 compiler bug on Hopper: `SharedEncodingAttr builder when the MMAEncodingAttr is Hopper has not been implemented yet`. Modal retried 3× before failing. Reverted to the hardcoded `(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)` config that produced the original **1.21× of FA-2 varlen** result. Autotuning is deferred to v0.1 pending a Triton 3.1+ image on Modal.

## [0.0.1] — 2026-05-24

First commit. Pre-alpha. The API may change between any two commits at this stage.

### Added

- `scree.Array` dataclass — packed `values + offsets + ragged_dim` representation with invariants enforced at construction
- `scree.pack`, `scree.unpack`, `scree.to_padded`, `scree.from_padded`, `scree.from_cu_seqlens` — the five core operations
- Backend dispatch for NumPy, PyTorch, and MLX (Apple Silicon, via Metal)
- `scree.bridges` — round-trip helpers for `torch.nested`, HuggingFace `(hidden_states, attention_mask)`, FlashAttention `cu_seqlens`, and cross-framework via DLPack
- `scree.kernels.reference` — slow but correct varlen kernels: `varlen_attention` (causal & non-causal), `varlen_layernorm`, `varlen_softmax`, all three working on NumPy, PyTorch, and MLX
- `scree.kernels.triton.varlen_attention_triton` — first-attempt FA-2 style varlen self-attention forward kernel for CUDA; correctness validated on H100 with max abs diff `4.88e-4` vs FlashAttention-2, timing **1.21× of FA-2 varlen** on a 12k-token workload
- Memory benchmark (`benchmarks/bench_memory.py`) reporting 71% mean savings on training-style batches and 85% on inference-style batches vs HuggingFace padded
- Modal-hosted H100 benchmark (`benchmarks/modal_bench.py`) for correctness + timing of the Triton kernel
- GitHub Actions CI on Python 3.10/3.11/3.12 (Ubuntu) and 3.11 (macOS); runs the full test suite, both examples, and the memory benchmark on every push
- Two end-to-end examples: a 6-line quickstart and a full pre-norm transformer block built only on scree primitives

### Test coverage
- 31 tests across NumPy + PyTorch + MLX
- 0 skipped on a machine with all three backends; partial skips when an optional backend is unavailable
