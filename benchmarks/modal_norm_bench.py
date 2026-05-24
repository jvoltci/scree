"""Modal: validate + time Triton RMSNorm and LayerNorm kernels on H100.

Compares scree's Triton kernels against PyTorch's native RMSNorm / LayerNorm
on a realistic packed-batch workload.

Run:
    modal run benchmarks/modal_norm_bench.py

Cost: ~$0.15 of Modal credit. One H100 allocation, ~2 minutes.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.4.0", "triton==3.0.0", "numpy", "packaging", "wheel", "ninja")
    .pip_install(FLASH_ATTN_WHEEL)
    .add_local_dir(str(REPO_ROOT / "src" / "scree"), "/root/scree_pkg/src/scree", copy=True)
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), "/root/scree_pkg/pyproject.toml", copy=True)
    .add_local_file(str(REPO_ROOT / "README.md"), "/root/scree_pkg/README.md", copy=True)
    .run_commands("pip install -e /root/scree_pkg")
)

app = modal.App("scree-norm-bench", image=image)


@app.function(gpu="H100", timeout=600)
def bench() -> dict:
    import time

    import torch

    from scree.kernels.triton import varlen_layernorm_triton, varlen_rmsnorm_triton

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(0)
    total = 12160
    feature_dim = 4096
    dtype = torch.float16

    x = torch.randn(total, feature_dim, dtype=dtype, device="cuda")
    weight = torch.randn(feature_dim, dtype=dtype, device="cuda")
    bias = torch.randn(feature_dim, dtype=dtype, device="cuda")

    print(f"workload: {total} tokens × {feature_dim} feature_dim, dtype={dtype}")
    print()

    # -------------------- RMSNorm --------------------
    # PyTorch reference (no native RMSNorm in torch 2.4; do it manually)
    def rms_ref():
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return ((x.float() / rms) * weight.float()).to(dtype)

    def rms_triton():
        return varlen_rmsnorm_triton(x, weight=weight, eps=1e-6)

    out_ref = rms_ref()
    out_tr = rms_triton()
    diff = (out_ref.float() - out_tr.float()).abs().max().item()
    print(f"RMSNorm correctness: max abs diff vs reference = {diff:.6e}")
    rms_pass = diff < 5e-3
    print(f"RMSNorm: {'PASS' if rms_pass else 'FAIL'}")

    def time_fn(fn, n_iter=100, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000  # ms

    ref_ms = time_fn(rms_ref)
    triton_ms = time_fn(rms_triton)
    print(f"RMSNorm timing:")
    print(f"  PyTorch reference: {ref_ms:.4f} ms")
    print(f"  scree-Triton:      {triton_ms:.4f} ms")
    print(f"  speedup:           {ref_ms / triton_ms:.2f}x")
    print()

    # -------------------- LayerNorm --------------------
    def ln_ref():
        return torch.nn.functional.layer_norm(x, (feature_dim,), weight, bias, eps=1e-5)

    def ln_triton():
        return varlen_layernorm_triton(x, weight=weight, bias=bias, eps=1e-5)

    out_ref = ln_ref()
    out_tr = ln_triton()
    diff = (out_ref.float() - out_tr.float()).abs().max().item()
    print(f"LayerNorm correctness: max abs diff vs torch.nn.functional.layer_norm = {diff:.6e}")
    ln_pass = diff < 5e-3
    print(f"LayerNorm: {'PASS' if ln_pass else 'FAIL'}")

    ref_ms = time_fn(ln_ref)
    triton_ms = time_fn(ln_triton)
    print(f"LayerNorm timing:")
    print(f"  torch.nn.functional.layer_norm: {ref_ms:.4f} ms")
    print(f"  scree-Triton:                   {triton_ms:.4f} ms")
    print(f"  speedup:                        {ref_ms / triton_ms:.2f}x")

    return {
        "rms_pass": rms_pass,
        "ln_pass": ln_pass,
    }


@app.local_entrypoint()
def main() -> None:
    print(bench.remote())
