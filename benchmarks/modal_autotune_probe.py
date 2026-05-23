"""Modal probe: identify Triton configs that compile and run on H100.

Triton 3.0 has a known compiler bug on Hopper for some BLOCK_M × BLOCK_N ×
num_warps combinations: ``SharedEncodingAttr builder when the
MMAEncodingAttr is Hopper has not been implemented yet``. This script
tests each candidate config in isolation, catches the SIGABRT/exception,
and returns a safe-list we can hardcode into the kernel's autotune grid.

Run:
    modal run benchmarks/modal_autotune_probe.py

Cost: ~$0.30-$0.80 of Modal credit. One H100 allocation, ~3-5 minutes
for the full 24-config sweep with timing on each successful config.
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
    .pip_install(
        "torch==2.4.0",
        "triton==3.0.0",
        "numpy",
        "packaging",
        "wheel",
        "ninja",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .add_local_dir(str(REPO_ROOT / "src" / "scree"), "/root/scree_pkg/src/scree", copy=True)
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), "/root/scree_pkg/pyproject.toml", copy=True)
    .add_local_file(str(REPO_ROOT / "README.md"), "/root/scree_pkg/README.md", copy=True)
    .run_commands("pip install -e /root/scree_pkg")
)

app = modal.App("scree-autotune-probe", image=image)


@app.function(gpu="H100", timeout=900)
def probe() -> dict:
    """Try every (BLOCK_M, BLOCK_N, num_warps, num_stages) and report which work."""
    import time
    import torch

    from scree.kernels.triton.varlen_attention import _varlen_attn_fwd_kernel

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    # Same workload as modal_bench.py — keeps the comparison apples-to-apples.
    torch.manual_seed(0)
    lengths = [512, 1024, 768, 256, 2048, 384, 896, 640, 1536, 320, 1152, 480, 704, 192, 832, 416]
    n_heads, head_dim = 16, 64
    dtype = torch.float16
    causal = True
    total = sum(lengths)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device="cuda").cumsum(0)
    max_seq = max(lengths)

    q = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total, n_heads, head_dim, dtype=dtype, device="cuda")

    import math
    import triton

    sm_scale = 1.0 / math.sqrt(head_dim)
    batch = cu_seqlens.numel() - 1

    def run_with(block_m: int, block_n: int, num_warps: int, num_stages: int) -> tuple[bool, float, str]:
        """Try one config. Returns (success, time_ms, error_msg)."""
        out = torch.empty_like(q)
        try:
            n_q_blocks = triton.cdiv(max_seq, block_m)
            grid = (batch, n_q_blocks, n_heads)

            # Warmup
            for _ in range(3):
                _varlen_attn_fwd_kernel[grid](
                    q, k, v, out, cu_seqlens, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    BLOCK_M=block_m, BLOCK_N=block_n,
                    HEAD_DIM=head_dim, CAUSAL=causal,
                    num_warps=num_warps, num_stages=num_stages,
                )
            torch.cuda.synchronize()

            # Time
            n_iter = 30
            t0 = time.perf_counter()
            for _ in range(n_iter):
                _varlen_attn_fwd_kernel[grid](
                    q, k, v, out, cu_seqlens, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    BLOCK_M=block_m, BLOCK_N=block_n,
                    HEAD_DIM=head_dim, CAUSAL=causal,
                    num_warps=num_warps, num_stages=num_stages,
                )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / n_iter * 1000
            return True, elapsed, ""
        except Exception as e:
            return False, float("inf"), str(e)[:120]

    configs = []
    for bm in (64, 128):
        for bn in (32, 64, 128):
            for nw in (4, 8):
                for ns in (2, 3):
                    configs.append((bm, bn, nw, ns))

    results = []
    safe_list = []
    print(f"probing {len(configs)} configs on workload "
          f"(total_tokens={total}, n_heads={n_heads}, head_dim={head_dim})")
    print()
    for cfg in configs:
        bm, bn, nw, ns = cfg
        ok, t_ms, err = run_with(bm, bn, nw, ns)
        if ok:
            print(f"  SAFE   BM={bm:3d} BN={bn:3d} warps={nw} stages={ns}  "
                  f"time={t_ms:6.3f} ms")
            safe_list.append({"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": nw, "num_stages": ns, "ms": t_ms})
        else:
            print(f"  UNSAFE BM={bm:3d} BN={bn:3d} warps={nw} stages={ns}  err={err[:70]}")
        results.append({"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": nw, "num_stages": ns,
                        "ok": ok, "ms": t_ms if ok else None, "err": err if not ok else ""})

    print()
    print(f"safe: {len(safe_list)} / {len(configs)}")
    if safe_list:
        best = min(safe_list, key=lambda c: c["ms"])
        print(f"fastest safe config: BM={best['BLOCK_M']} BN={best['BLOCK_N']} "
              f"warps={best['num_warps']} stages={best['num_stages']}  "
              f"time={best['ms']:.3f} ms")
        # Sort safe by time, print top 5
        ranked = sorted(safe_list, key=lambda c: c["ms"])[:5]
        print()
        print(f"top {len(ranked)} safe configs by time:")
        for r in ranked:
            print(f"  BM={r['BLOCK_M']:3d} BN={r['BLOCK_N']:3d} warps={r['num_warps']} "
                  f"stages={r['num_stages']}  {r['ms']:.3f} ms")

    return {"results": results, "safe": safe_list}


@app.local_entrypoint()
def main() -> None:
    result = probe.remote()
    print()
    print("== summary ==")
    print(f"  total configs probed: {len(result['results'])}")
    print(f"  safe: {len(result['safe'])}")
    if result["safe"]:
        ranked = sorted(result["safe"], key=lambda c: c["ms"])
        print(f"  fastest: BM={ranked[0]['BLOCK_M']} BN={ranked[0]['BLOCK_N']} "
              f"warps={ranked[0]['num_warps']} stages={ranked[0]['num_stages']}  "
              f"{ranked[0]['ms']:.3f} ms")
