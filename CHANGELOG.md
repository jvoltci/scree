# Changelog

All notable changes to scree are recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Comprehensive documentation under `docs/` covering concepts, API reference, bridges, kernels, architecture, benchmarks, and FAQ
- `CONTRIBUTING.md` and `CHANGELOG.md`
- `varlen_rmsnorm` reference kernel — the norm used by LLaMA / Mistral / Mixtral / DeepSeek / Qwen and most modern open transformers (replaces LayerNorm in nearly every post-2023 architecture)
- `benchmarks/bench_throughput.py` — CPU throughput benchmark. **6.6× faster than padded baseline** on `varlen_attention` for a realistic 16-sequence batch.
- **JAX backend.** All four reference kernels (varlen_attention, varlen_layernorm, varlen_rmsnorm, varlen_softmax) now run on JAX arrays. Cross-backend numerical agreement vs NumPy reference (atol 1e-4 for matmul-heavy ops, 1e-5 otherwise). Closes the v0.1 "cross-framework" claim: NumPy + PyTorch + MLX + JAX all pass the same correctness suite.
- **Autograd verification.** `examples/03_train_step.py` demonstrates a real training loop with PyTorch autograd flowing through `scree.Array` — loss drops 80× over 30 steps on a synthetic copy task. `tests/test_autograd.py` locks this in with focused tests: gradient flow through pack/unpack, varlen_attention (q/k/v gradients), varlen_layernorm + varlen_rmsnorm (weight/bias gradients), and the length-1 edge case.
- **HuggingFace migration recipe.** `examples/04_hf_compat.py` walks through the migration pattern any HF user follows: `(hidden_states, attention_mask)` → `bridges.from_hf_padded` → scree-native op → `bridges.to_hf_padded` → bit-exact match with the HF-native implementation. No `transformers` install required — the example simulates the HF interface.
- **Property-based tests via Hypothesis.** `tests/test_properties.py` adds 11 generative tests for the invariants and operation properties: pack/unpack and to_padded/from_padded roundtrip identity, offsets monotonicity, lengths consistency, varlen_softmax row-sum=1, varlen_layernorm zero-mean unit-var, varlen_rmsnorm unit-rms, varlen_attention shape preservation, and zero-copy `from_cu_seqlens`. Each property runs across many randomly-generated batches, catching edge cases the targeted unit tests would miss. Test count 31 → 56.
- **Triton autograd wrapper + full Triton backward kernels.** `scree.kernels.triton.varlen_attention_triton_autograd` is a `torch.autograd.Function` that runs both forward AND backward on Triton kernels — no slow Python path anymore. The backward set is FA-2 style:
  - `_varlen_attn_bwd_preprocess_kernel` — computes per-token `Delta = sum(O * dO)`
  - `_varlen_attn_bwd_dkv_kernel` — accumulates dK, dV per K/V tile
  - `_varlen_attn_bwd_dq_kernel` — accumulates dQ per Q tile
  - The forward kernel was extended to save LSE (log-sum-exp) per (token, head) so backward can recompute attention probs without materializing the full matrix.

  Verified on H100 (16 sequences × log-normal lengths, 12160 total tokens, 16 heads × head_dim 64, fp16, causal):
  - Forward correctness PASS (max abs diff 4.88e-04 vs FA-2)
  - Backward correctness PASS (dq 9.77e-04, dk 1.95e-03, dv 1.95e-03 vs FA-2)
  - Headline workload: forward **1.30×** of FA-2 (added LSE save vs the original 1.21×), full training step **1.61×** of FA-2 (down from 22× with the reference backward — a ~14× improvement).
- **Subprocess-isolated autotune probe.** `benchmarks/modal_autotune_probe_isolated.py` maps the full safe-set on H100 by running each candidate config in a subprocess (SIGABRT crashes the child, parent keeps going). Result: **18 of 24 configs are safe**; all 6 unsafe configs share the pattern `BM=64 × num_warps=8`. Fastest config measured kernel-only at 0.160 ms — slightly faster than FA-2's 0.165 ms (but wrapper allocation overhead brings the user-facing forward to 0.216 ms = 1.30×). The full safe-list ships as a code comment for future autotune work.
- **Multi-shape benchmark sweep.** `benchmarks/modal_multishape_sweep.py` characterizes scree-Triton vs FA-2 across 27 shapes (head_dim × n_heads × mean_len = 3 × 3 × 3). Median forward ratio 1.95×, median training-step ratio 2.01×. **scree is closer to parity on large workloads**, slower on toy ones because wrapper allocation overhead is per-call. Best case: 1.21× forward / 1.45× training-step at (head_dim=64, n_heads=16, mean_len=2048).
- **GitHub Pages docs deployment.** `mkdocs.yml` + `.github/workflows/docs.yml` auto-deploy the `docs/` tree to `jvoltci.github.io/scree` on every push to main. Material theme, navigation matching the docs structure, light/dark palette, code-copy buttons.
- **RELEASE.md.** v0.1.0 release-readiness checklist covering code, API, docs, benchmarks, packaging, pre-launch credibility, repo hygiene, release artifacts, plus a launch-day sequence and hotfix flow.
- **v0.1 launch blog draft** — written and reviewed pre-launch (kept off the public repo until the launch is live).
- **Buffer-reuse on `varlen_attention_triton`** — `out=None, lse_buffer=None` kwargs let users skip the ~0.05 ms per-call allocation in hot loops.
- **Triton RMSNorm + LayerNorm kernels** (`varlen_rmsnorm_triton`, `varlen_layernorm_triton`). H100, fp16, 12160 × 4096:
  - RMSNorm: **13.97×** speedup vs PyTorch reference (no native), PASS correctness
  - LayerNorm: **1.31×** speedup vs `torch.nn.functional.layer_norm`, PASS correctness
- **`examples/05_multimodal_interleaved.py`** — text + image-patch interleaving in one scree.Array.
- **`docs/vllm-integration-sketch.md`** — architectural sketch of the vLLM/SGLang integration proposal.
- **Array API conformance tests** (`tests/test_array_api.py`, 10 tests) — verifies the Array API contract on `scree.Array.values`.
- **Cross-backend property tests** — 2 new hypothesis tests that exercise the invariants across NumPy + PyTorch + MLX + JAX in a single run.

### Investigated (deferred)
- Triton autotune (3-config safe grid + autotune key=[]) ran 1.50× of FA-2 — worse than the hardcoded 1.21×. Autotune overhead appears to leak into per-iteration timing for short kernels, or the autotune picked a config that benchmarks fast in isolation but loses in the full bench loop. Reverted to hardcoded `(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)`. Probed safe configs documented in code comment for future re-evaluation.
- `benchmarks/modal_autotune_probe.py` runs each config in series but the Triton 3.0 Hopper bug crashes the whole container, forcing Modal to retry. The subprocess-isolated version (`modal_autotune_probe_isolated.py`) maps the full 18-safe / 6-unsafe split.
- Per-shape autotune dispatch — the multi-shape sweep was N=1 per shape, not enough data to confidently route configs by `(head_dim, n_heads)`. The 5–10% potential gains don't justify the complexity in v0.0. Revisit when shape-specific perf becomes a bottleneck.

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
