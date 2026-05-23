# Benchmarks

Methodology, current numbers, and how to reproduce them.

## Memory savings vs HF padded (CPU, runs anywhere)

Script: [`benchmarks/bench_memory.py`](../benchmarks/bench_memory.py)

### Methodology

The script samples sequence lengths from a log-normal distribution (the empirical shape of real LLM batches — most prompts are short, a few are long). For each batch:

1. Generate `num_seqs` random sequences with those lengths × `feature_dim` features in fp32.
2. Pack into a `scree.Array` and measure `values.nbytes + offsets.nbytes`.
3. Convert to HF padded form via `to_hf_padded(arr)` and measure `padded.nbytes + mask.nbytes`.
4. Report the savings ratio.

Five trials with different seeds; mean, median, min, max reported.

The choice of fp32 is conservative — the absolute byte counts scale, but the *ratio* (scree vs padded) depends only on `valid_tokens / padded_tokens`, which is dtype-independent.

### Current numbers

| Workload | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- |
| `--num-seqs 64 --mean-len 256 --sigma 0.6 --feature-dim 4096` (training) | 71% | 70% | 63% | 84% |
| `--num-seqs 32 --mean-len 1024 --sigma 1.2 --feature-dim 4096` (inference) | 85% | 87% | 75% | 94% |

### Reproduce

```bash
# Default (training-style):
python benchmarks/bench_memory.py

# Inference-style with longer tail:
python benchmarks/bench_memory.py --num-seqs 32 --mean-len 1024 --sigma 1.2 --feature-dim 4096

# Custom:
python benchmarks/bench_memory.py --num-seqs 128 --feature-dim 8192 --mean-len 512 --trials 10
```

Output is plain text with per-trial numbers and an aggregate summary. Saved to stdout — pipe to a file if you want to keep it.

### What this number means

The ratio represents the percentage of memory the padded representation wastes on padding tokens that aren't real data. With 85% savings, the padded representation is 6.5× larger than the scree representation in memory.

This is a **lower bound on FLOPs savings** during training. If your transformer block does any operation that scales linearly with sequence position (e.g., layernorm, feedforward, even non-attention attention paths), those FLOPs are wasted on padding in the padded representation. scree skips them by construction.

For attention specifically, the FLOPs ratio depends on the kernel: FlashAttention varlen already skips padding even when called with `attention_mask`, but the HF default attention path does not.

## GPU kernel benchmark: scree-Triton vs FlashAttention-2 varlen (cloud H100)

Script: [`benchmarks/modal_bench.py`](../benchmarks/modal_bench.py)

### Methodology

Runs on a single NVIDIA H100 80GB SXM via [Modal](https://modal.com/). The script:

1. Generates a 16-sequence batch with a fixed length distribution `[512, 1024, 768, 256, 2048, 384, 896, 640, 1536, 320, 1152, 480, 704, 192, 832, 416]` (mean ~760, max 2048, 16 heads × head_dim 64, fp16, causal).
2. Calls both `flash_attn_varlen_func` and `scree.kernels.triton.varlen_attention_triton` on identical input.
3. Compares element-wise: `max abs diff < 5e-3` is required to pass correctness.
4. Times each kernel: 10 warmup iterations followed by 50 measured iterations. Uses `torch.cuda.synchronize()` before timing.
5. Reports the per-iteration mean (ms) for each kernel and the ratio.

The workload (12,160 total tokens) is chosen to be representative of a single training step or inference forward — not so small that launch overhead dominates, not so large that HBM bandwidth saturates.

### Current numbers

NVIDIA H100 80GB SXM, fp16, causal, 12,160 total tokens, 16 heads × head_dim 64:

| Kernel | Time | Ratio |
| --- | --- | --- |
| FlashAttention-2 varlen | 0.166 ms | 1.00× |
| scree-Triton varlen     | 0.201 ms | 1.21× |

Correctness: `max abs diff = 4.88e-4` vs FA-2 → PASS.

This is the first measured run (commit `12d7579`). The v0.0 → v0.1 milestone target is **≤1.20× with PASS correctness** — we hit 1.21×, narrowly above target.

### Reproduce

```bash
# One-time setup:
pip install modal
modal token new            # auth via browser

# Run the benchmark:
modal run benchmarks/modal_bench.py
```

First run: ~5-10 min for image build (PyTorch + Triton + flash-attn + scree). Subsequent runs: ~2-5 min (image cached).

Cost: ~$0.15 – $0.40 per run. The function is `timeout=900` (15 min wall-clock cap) so cost can't run away.

### What this number means

scree's Triton kernel is at parity (within ~20%) with the best public varlen attention implementation. With autotuning enabled and bf16 specialization, the kernel should match or beat FA-2 on the next iteration.

The kernel is not better than FA-2 because FA-2 is a heavily tuned production codebase from Tri Dao's team that's had two years of refinement. Matching it on first attempt is the realistic goal; beating it is not.

### Why this matters

If scree's kernel were 3-5× slower than FA-2, the project's headline claim ("the cross-framework primitive with reference-quality kernels") would be false. The kernel needs to be fast enough that users don't pay a penalty for going through scree's abstraction. 1.21× is *acceptable*; sub-1.0× is *delightful*. The goal between v0.0 and v0.1 is to get to 1.0× or below.

## What's NOT benchmarked yet

These will be added before the v0.1 launch:

- **End-to-end training:** scree-native training of a 1B-parameter transformer vs HF padded baseline. Target: bit-identical loss curve at ≥1.3× tokens/sec.
- **End-to-end inference:** vLLM-style continuous batching with scree vs vLLM-native. Target: parity throughput, lower memory.
- **Cross-framework:** numerical agreement between PyTorch backend and JAX backend for `varlen_attention` (JAX backend pending).
- **Backward pass:** scree-Triton backward vs FA-2 backward. Currently scree forward only.

These are tracked in the v0.1 roadmap in [architecture.md](architecture.md).

## Cost guard

Modal benchmarks are wrapped in `@app.function(timeout=900)` (15-minute hard cap). Even a pathological infinite-loop bug can't burn more than ~$1 of credit.

For long-running sweeps (autotune grid, multi-shape benchmarks), the timeout will be bumped — but the cost guard always exists.

## Calibration runs

If you want to validate scree's numbers on your own hardware:

```bash
# CPU memory benchmark on your laptop:
python benchmarks/bench_memory.py

# GPU kernel benchmark on your CUDA box (skip Modal):
python -c "
import torch
from scree.kernels.triton import varlen_attention_triton
# ... follow the modal_bench.py recipe with your own workload
"
```

If your numbers disagree materially with what's reported above, please open an issue with your hardware (`nvidia-smi`), driver, CUDA version, and Triton version. We want the benchmark numbers to be honest.
