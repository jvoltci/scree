# Benchmarks

## bench_memory.py (CPU, runs anywhere)

Measures exact memory footprint of `scree.Array` (packed values+offsets) vs
HuggingFace `(hidden_states, attention_mask)` on realistic log-normal LLM
length distributions.

```bash
python benchmarks/bench_memory.py
python benchmarks/bench_memory.py --num-seqs 128 --feature-dim 4096 --mean-len 512
```

Current headline (CPU, no GPU needed):

| Workload | Mean savings |
| --- | --- |
| Training-style (batch 64, mean_len 256) | **71%** |
| Inference-style (batch 32, mean_len 1024) | **85%** |

## modal_bench.py (cloud H100 via Modal)

Validates the scree-Triton `varlen_attention` kernel against FlashAttention-2
varlen on a real H100, then times both.

**Prerequisites**

```bash
pip install modal
modal token new   # one-time auth
```

**Run**

```bash
modal run benchmarks/modal_bench.py
```

First run builds the image (PyTorch + Triton + flash-attn) — ~10 minutes.
Subsequent runs reuse the cached image and complete in 2–5 minutes.

**Cost guard**

The Modal function has a `timeout=900` (15 min wall-clock cap) to prevent
runaway charges. H100 on Modal is approximately $4/hr, so a typical
benchmark run consumes **$0.15 – $0.40** of credit.

**What it does**

1. Allocates one H100 instance
2. Runs both `flash_attn_varlen_func` and `scree.kernels.triton.varlen_attention_triton`
   on the same input
3. Compares output element-wise (max absolute diff must be < 5e-3 for fp16)
4. Times both kernels averaged over 50 iterations after 10 warmup
5. Prints the ratio (scree / FA — lower is better; 1.00x = parity)
6. Shuts the container down

**Expected output (target for v0.1)**

```
correctness: max abs diff = ~1e-3, rel = ~5e-4
correctness: PASS

FlashAttention-2 varlen:   X.XXX ms
scree-Triton varlen:       Y.YYY ms
scree / FA ratio:          ~1.2x  (lower is better)
```

The v0.1 milestone is **scree / FA ratio ≤ 1.2x with PASS correctness**.
If the first run is much slower than that (e.g. 3–5×), it's expected —
the kernel ships unautotuned in v0.0. Tuning (`block_m`, `block_n`,
`num_warps`, `num_stages`) is the v0.0 → v0.1 work.
