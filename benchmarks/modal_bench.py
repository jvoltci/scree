"""Modal-hosted benchmark: scree-Triton varlen_attention vs FlashAttention-2.

Spins up a single H100 instance, installs PyTorch + Triton + flash-attn,
mounts the local scree source, runs a correctness check (scree-Triton
output vs FlashAttention-2 output), then times both kernels on a
realistic varlen workload. Total runtime ~3–8 minutes after first build.

Usage
-----
First time (one-time setup, ~10 min for image build):
    pip install modal
    modal token new                          # auth
    modal run benchmarks/modal_bench.py

Subsequent runs (image cached):
    modal run benchmarks/modal_bench.py

Cost
----
H100 on Modal is ~$0.001/sec ≈ $4/hr. A typical run uses 2–5 minutes
of H100 time → $0.15 – $0.40 per run.

Cost guard: ``@app.function(timeout=900)`` caps a single run at 15
minutes wall clock to prevent runaway charges.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "triton==3.0.0",
        "numpy",
        "packaging",
        "wheel",
        "ninja",
    )
    # flash-attn requires --no-build-isolation so it sees torch/ninja from the env
    .pip_install("flash-attn==2.6.3", extra_options="--no-build-isolation")
    # Mount the scree package so `import scree` works on the GPU container.
    .add_local_dir(str(REPO_ROOT / "src" / "scree"), "/root/scree_pkg/scree", copy=True)
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), "/root/scree_pkg/pyproject.toml", copy=True)
    .add_local_file(str(REPO_ROOT / "README.md"), "/root/scree_pkg/README.md", copy=True)
    .run_commands("pip install -e /root/scree_pkg")
)

app = modal.App("scree-bench", image=image)


@app.function(gpu="H100", timeout=900)
def bench() -> dict:
    """Run on H100: correctness check + timing for scree-Triton vs FA-2."""
    import time

    import torch
    from flash_attn import flash_attn_varlen_func

    from scree.kernels.triton import varlen_attention_triton

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    # Realistic varlen workload: 16 sequences with a log-normal length
    # distribution centered around 1024 tokens.
    torch.manual_seed(0)
    lengths = [512, 1024, 768, 256, 2048, 384, 896, 640, 1536, 320, 1152, 480, 704, 192, 832, 416]
    n_heads = 16
    head_dim = 64
    dtype = torch.float16

    total = sum(lengths)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device="cuda").cumsum(0)
    max_seq = max(lengths)

    print(f"workload: {len(lengths)} sequences, {total} total tokens, max_seq={max_seq}")
    print(f"heads={n_heads}, head_dim={head_dim}, dtype={dtype}")
    print()

    q = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")

    # Correctness — FlashAttention-2 is the ground truth.
    def run_fa() -> torch.Tensor:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seq, max_seq, causal=True
        )

    def run_scree() -> torch.Tensor:
        return varlen_attention_triton(q, k, v, cu_seqlens, causal=True)

    out_fa = run_fa()
    out_scree = run_scree()
    max_diff = (out_fa.float() - out_scree.float()).abs().max().item()
    rel_diff = (max_diff / out_fa.float().abs().max().item()) if out_fa.numel() else 0.0
    print(f"correctness: max abs diff = {max_diff:.6e}, rel = {rel_diff:.6e}")
    correct = max_diff < 5e-3  # generous fp16 tolerance
    print(f"correctness: {'PASS' if correct else 'FAIL'}")
    print()

    # Timing — warmup then average.
    def time_fn(fn, n_iter: int = 50, warmup: int = 10) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000  # ms

    fa_ms = time_fn(run_fa)
    scree_ms = time_fn(run_scree)
    ratio = scree_ms / fa_ms if fa_ms > 0 else float("inf")

    print(f"FlashAttention-2 varlen: {fa_ms:7.3f} ms")
    print(f"scree-Triton varlen:     {scree_ms:7.3f} ms")
    print(f"scree / FA ratio:        {ratio:5.2f}x  (lower is better; 1.00x = parity)")

    return {
        "correctness_pass": correct,
        "max_abs_diff": max_diff,
        "fa_ms": fa_ms,
        "scree_ms": scree_ms,
        "ratio": ratio,
        "total_tokens": total,
        "n_heads": n_heads,
        "head_dim": head_dim,
    }


@app.local_entrypoint()
def main() -> None:
    result = bench.remote()
    print()
    print("== summary ==")
    for key, value in result.items():
        print(f"  {key}: {value}")
