"""Modal: characterize scree-Triton vs FA-2 across realistic shapes.

Sweeps (head_dim, n_heads, length-distribution) and reports
forward-only and training-step (forward+backward) ratios. Useful for:

  - README headline tables that aren't tied to a single workload
  - identifying which shapes scree wins / loses on
  - finding regressions when the kernel changes

Run:
    modal run benchmarks/modal_multishape_sweep.py

Cost: ~$0.30-$0.60 of Modal credit. Each shape times ~25 iters
forward-only and ~25 iters forward+backward; 27 shapes ≈ 4-6 minutes.
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

app = modal.App("scree-multishape-sweep", image=image)


@app.function(gpu="H100", timeout=900)
def sweep() -> list:
    """Forward and training-step timing across 27 shapes."""
    import time

    import numpy as np
    import torch
    from flash_attn import flash_attn_varlen_func

    from scree.kernels.triton import varlen_attention_triton, varlen_attention_triton_autograd

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    def lengths_for(num_seqs: int, mean: int, sigma: float = 0.6, seed: int = 0):
        rng = np.random.default_rng(seed)
        l = rng.lognormal(mean=np.log(mean), sigma=sigma, size=num_seqs)
        return [max(1, int(x)) for x in l]

    num_seqs = 16
    shapes = []
    for head_dim in (32, 64, 128):
        for n_heads in (4, 8, 16):
            for mean_len in (256, 1024, 2048):
                shapes.append((head_dim, n_heads, mean_len))

    print(f"sweep: {len(shapes)} shapes  (head_dim × n_heads × mean_len)")
    print(f"       num_seqs=16, sigma=0.6, fp16, causal=True")
    print()
    print(f"{'head_dim':>8} {'n_heads':>7} {'mean_len':>8}  "
          f"{'fa_fwd':>9} {'sc_fwd':>9} {'fwd_x':>6}  "
          f"{'fa_step':>9} {'sc_step':>9} {'step_x':>6}")
    print("-" * 95)

    results = []
    for head_dim, n_heads, mean_len in shapes:
        torch.manual_seed(0)
        lengths = lengths_for(num_seqs, mean_len, sigma=0.6, seed=0)
        total = sum(lengths)
        max_seq = max(lengths)
        cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device="cuda")
        cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device="cuda").cumsum(0)

        def make_qkv(grad: bool):
            q = torch.randn(total, n_heads, head_dim, dtype=torch.float16, device="cuda")
            k = torch.randn(total, n_heads, head_dim, dtype=torch.float16, device="cuda")
            v = torch.randn(total, n_heads, head_dim, dtype=torch.float16, device="cuda")
            if grad:
                q.requires_grad_(True)
                k.requires_grad_(True)
                v.requires_grad_(True)
            return q, k, v

        # --- forward only ---
        q, k, v = make_qkv(grad=False)

        def fa_fwd():
            return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seq, max_seq, causal=True)

        def sc_fwd():
            return varlen_attention_triton(q, k, v, cu_seqlens, causal=True)

        # warmup
        for _ in range(5):
            fa_fwd()
            sc_fwd()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(25):
            fa_fwd()
        torch.cuda.synchronize()
        fa_fwd_ms = (time.perf_counter() - t0) / 25 * 1000

        t0 = time.perf_counter()
        for _ in range(25):
            sc_fwd()
        torch.cuda.synchronize()
        sc_fwd_ms = (time.perf_counter() - t0) / 25 * 1000

        # --- training step (forward + backward) ---
        def fa_step():
            q, k, v = make_qkv(grad=True)
            o = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seq, max_seq, causal=True)
            o.sum().backward()

        def sc_step():
            q, k, v = make_qkv(grad=True)
            o = varlen_attention_triton_autograd(q, k, v, cu_seqlens, causal=True)
            o.sum().backward()

        for _ in range(3):
            fa_step()
            sc_step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(20):
            fa_step()
        torch.cuda.synchronize()
        fa_step_ms = (time.perf_counter() - t0) / 20 * 1000

        t0 = time.perf_counter()
        for _ in range(20):
            sc_step()
        torch.cuda.synchronize()
        sc_step_ms = (time.perf_counter() - t0) / 20 * 1000

        fwd_ratio = sc_fwd_ms / fa_fwd_ms
        step_ratio = sc_step_ms / fa_step_ms
        print(f"{head_dim:>8} {n_heads:>7} {mean_len:>8}  "
              f"{fa_fwd_ms:>7.3f}ms {sc_fwd_ms:>7.3f}ms {fwd_ratio:>5.2f}x  "
              f"{fa_step_ms:>7.3f}ms {sc_step_ms:>7.3f}ms {step_ratio:>5.2f}x")
        results.append({
            "head_dim": head_dim, "n_heads": n_heads, "mean_len": mean_len,
            "total_tokens": total,
            "fa_fwd_ms": fa_fwd_ms, "sc_fwd_ms": sc_fwd_ms, "fwd_ratio": fwd_ratio,
            "fa_step_ms": fa_step_ms, "sc_step_ms": sc_step_ms, "step_ratio": step_ratio,
        })

    print()
    print("=" * 50)
    fwd_ratios = [r["fwd_ratio"] for r in results]
    step_ratios = [r["step_ratio"] for r in results]
    print(f"forward-only ratio (scree / FA-2):")
    print(f"  median = {sorted(fwd_ratios)[len(fwd_ratios) // 2]:.2f}x")
    print(f"  min    = {min(fwd_ratios):.2f}x  ({sorted(results, key=lambda r: r['fwd_ratio'])[0]})")
    print(f"  max    = {max(fwd_ratios):.2f}x  ({sorted(results, key=lambda r: r['fwd_ratio'])[-1]})")
    print()
    print(f"full training step ratio (scree / FA-2):")
    print(f"  median = {sorted(step_ratios)[len(step_ratios) // 2]:.2f}x")
    print(f"  min    = {min(step_ratios):.2f}x")
    print(f"  max    = {max(step_ratios):.2f}x")

    return results


@app.local_entrypoint()
def main() -> None:
    results = sweep.remote()
    print()
    print(f"== summary ==")
    print(f"  total shapes:        {len(results)}")
    fwd = [r["fwd_ratio"] for r in results]
    step = [r["step_ratio"] for r in results]
    print(f"  forward median:      {sorted(fwd)[len(fwd) // 2]:.2f}x")
    print(f"  training step median: {sorted(step)[len(step) // 2]:.2f}x")
