"""Modal: verify and time the autograd wrapper around varlen_attention_triton.

Tests three things on H100:
  1. Forward correctness — scree-Triton output vs FlashAttention-2 output
  2. Backward correctness — scree-Triton-autograd gradients vs FA-2 gradients
  3. Forward + backward timing — scree vs FA-2

The autograd wrapper currently uses the Triton kernel for forward and the
reference (PyTorch ops) for backward. The reference backward is correct
but slow; the FA-2 Triton backward implementation lands in v0.1.

Run:
    modal run benchmarks/modal_autograd_bench.py

Cost: ~$0.30-$0.60 of Modal credit. One H100 allocation, ~3-5 minutes.
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

app = modal.App("scree-autograd-bench", image=image)


@app.function(gpu="H100", timeout=900)
def bench() -> dict:
    """Forward + backward correctness + timing on H100."""
    import time

    import torch
    from flash_attn import flash_attn_varlen_func

    from scree.kernels.triton import varlen_attention_triton_autograd

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(0)
    lengths = [512, 1024, 768, 256, 2048, 384, 896, 640, 1536, 320, 1152, 480, 704, 192, 832, 416]
    n_heads, head_dim = 16, 64
    dtype = torch.float16

    total = sum(lengths)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device="cuda").cumsum(0)
    max_seq = max(lengths)

    print(f"workload: {len(lengths)} sequences, {total} total tokens, max_seq={max_seq}")
    print(f"heads={n_heads}, head_dim={head_dim}, dtype={dtype}, causal=True")
    print()

    # Independent tensors for each path so backward gradients are independent.
    base_q = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")
    base_k = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")
    base_v = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")

    def make_inputs():
        return (
            base_q.detach().clone().requires_grad_(True),
            base_k.detach().clone().requires_grad_(True),
            base_v.detach().clone().requires_grad_(True),
        )

    # --- Forward correctness ---
    q_fa, k_fa, v_fa = make_inputs()
    q_sc, k_sc, v_sc = make_inputs()
    out_fa = flash_attn_varlen_func(q_fa, k_fa, v_fa, cu_seqlens, cu_seqlens, max_seq, max_seq, causal=True)
    out_sc = varlen_attention_triton_autograd(q_sc, k_sc, v_sc, cu_seqlens, causal=True)
    fwd_max_diff = (out_fa.float() - out_sc.float()).abs().max().item()
    print(f"forward correctness:  max abs diff = {fwd_max_diff:.6e}")
    fwd_pass = fwd_max_diff < 5e-3
    print(f"forward:              {'PASS' if fwd_pass else 'FAIL'}")
    print()

    # --- Backward correctness ---
    grad_out = torch.randn_like(out_fa)
    out_fa.backward(grad_out)
    out_sc.backward(grad_out)
    dq_diff = (q_fa.grad.float() - q_sc.grad.float()).abs().max().item()
    dk_diff = (k_fa.grad.float() - k_sc.grad.float()).abs().max().item()
    dv_diff = (v_fa.grad.float() - v_sc.grad.float()).abs().max().item()
    print(f"backward correctness:")
    print(f"  dq max abs diff = {dq_diff:.6e}")
    print(f"  dk max abs diff = {dk_diff:.6e}")
    print(f"  dv max abs diff = {dv_diff:.6e}")
    bwd_pass = max(dq_diff, dk_diff, dv_diff) < 5e-2  # looser bound: backward accumulates error
    print(f"backward:             {'PASS' if bwd_pass else 'FAIL'}")
    print()

    # --- Timing: forward + backward ---
    def time_fa(n_iter=30, warmup=5):
        for _ in range(warmup):
            q, k, v = make_inputs()
            out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seq, max_seq, causal=True)
            out.sum().backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            q, k, v = make_inputs()
            out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seq, max_seq, causal=True)
            out.sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000

    def time_sc(n_iter=30, warmup=5):
        for _ in range(warmup):
            q, k, v = make_inputs()
            out = varlen_attention_triton_autograd(q, k, v, cu_seqlens, causal=True)
            out.sum().backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            q, k, v = make_inputs()
            out = varlen_attention_triton_autograd(q, k, v, cu_seqlens, causal=True)
            out.sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000

    fa_ms = time_fa()
    sc_ms = time_sc()
    print(f"forward+backward timing (full training step):")
    print(f"  FlashAttention-2 varlen:    {fa_ms:7.3f} ms")
    print(f"  scree-Triton-autograd:      {sc_ms:7.3f} ms")
    print(f"  ratio (sc / fa):            {sc_ms / fa_ms:5.2f}x   (lower is better)")
    print()
    print("scree backward uses the full Triton backward kernel set")
    print("(preprocess + dKV + dQ, FA-2 style). Both forward and backward")
    print("are GPU-native Triton paths.")

    return {
        "forward_pass": fwd_pass,
        "backward_pass": bwd_pass,
        "fwd_max_diff": fwd_max_diff,
        "dq_max_diff": dq_diff,
        "dk_max_diff": dk_diff,
        "dv_max_diff": dv_diff,
        "fa_ms": fa_ms,
        "sc_ms": sc_ms,
        "ratio": sc_ms / fa_ms,
    }


@app.local_entrypoint()
def main() -> None:
    result = bench.remote()
    print()
    print("== summary ==")
    for key, value in result.items():
        print(f"  {key}: {value}")
