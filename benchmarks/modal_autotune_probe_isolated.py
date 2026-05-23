"""Modal probe with subprocess isolation per config.

The naive probe (modal_autotune_probe.py) crashes on the first
SIGABRT-causing Triton config because the abort kills the whole
container. This version runs each config in a subprocess so the crash
is contained — one Modal H100 allocation can map the full safe-set.

Run:
    modal run benchmarks/modal_autotune_probe_isolated.py

Cost: ~$0.50 of Modal credit. One H100 allocation, ~5-8 minutes for 24
configs (a few seconds per config; failed configs time out fast).
"""

from __future__ import annotations

import json
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

app = modal.App("scree-autotune-probe-isolated", image=image)


# Child-process script: load the kernel, run with one config, print timing as JSON.
CHILD_SCRIPT = r"""
import sys, json, time, math
import torch, triton
from scree.kernels.triton.varlen_attention import _varlen_attn_fwd_kernel

bm, bn, nw, ns = {bm}, {bn}, {nw}, {ns}

torch.manual_seed(0)
lengths = [512, 1024, 768, 256, 2048, 384, 896, 640, 1536, 320, 1152, 480, 704, 192, 832, 416]
n_heads, head_dim = 16, 64
total = sum(lengths); max_seq = max(lengths); batch = len(lengths)
cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device="cuda").cumsum(0)

q = torch.randn(total, n_heads, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(total, n_heads, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(total, n_heads, head_dim, dtype=torch.float16, device="cuda")
out = torch.empty_like(q)
lse = torch.empty((total, n_heads), dtype=torch.float32, device="cuda")
sm_scale = 1.0 / math.sqrt(head_dim)

n_q_blocks = triton.cdiv(max_seq, bm)
grid = (batch, n_q_blocks, n_heads)

# Warmup + time
for _ in range(5):
    _varlen_attn_fwd_kernel[grid](
        q, k, v, out, lse, cu_seqlens, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=head_dim, CAUSAL=True,
        num_warps=nw, num_stages=ns,
    )
torch.cuda.synchronize()
t0 = time.perf_counter()
n_iter = 50
for _ in range(n_iter):
    _varlen_attn_fwd_kernel[grid](
        q, k, v, out, lse, cu_seqlens, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=head_dim, CAUSAL=True,
        num_warps=nw, num_stages=ns,
    )
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / n_iter * 1000
print("RESULT_JSON:" + json.dumps({{"ms": ms, "bm": bm, "bn": bn, "nw": nw, "ns": ns}}))
"""


@app.function(gpu="H100", timeout=1200)
def probe_all() -> dict:
    """Probe every (BLOCK_M, BLOCK_N, num_warps, num_stages); subprocess each."""
    import subprocess

    configs = []
    for bm in (64, 128):
        for bn in (32, 64, 128):
            for nw in (4, 8):
                for ns in (2, 3):
                    configs.append((bm, bn, nw, ns))

    print(f"probing {len(configs)} configs (each in a subprocess)")
    print()

    safe = []
    unsafe = []
    for bm, bn, nw, ns in configs:
        script = CHILD_SCRIPT.format(bm=bm, bn=bn, nw=nw, ns=ns)
        try:
            result = subprocess.run(
                ["python", "-c", script],
                capture_output=True,
                timeout=90,
                text=True,
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT  BM={bm:3d} BN={bn:3d} warps={nw} stages={ns}")
            unsafe.append({"bm": bm, "bn": bn, "nw": nw, "ns": ns, "err": "timeout"})
            continue

        if result.returncode == 0:
            # Find the RESULT_JSON line
            payload = None
            for line in result.stdout.splitlines():
                if line.startswith("RESULT_JSON:"):
                    payload = json.loads(line[len("RESULT_JSON:"):])
                    break
            if payload is None:
                print(f"  WEIRD    BM={bm:3d} BN={bn:3d} warps={nw} stages={ns}  no result line")
                unsafe.append({"bm": bm, "bn": bn, "nw": nw, "ns": ns, "err": "no result"})
            else:
                print(f"  SAFE     BM={bm:3d} BN={bn:3d} warps={nw} stages={ns}  time={payload['ms']:7.4f} ms")
                safe.append(payload)
        else:
            stderr_tail = result.stderr.strip().splitlines()[-3:] if result.stderr else []
            err_msg = " | ".join(stderr_tail)[:120]
            print(f"  UNSAFE   BM={bm:3d} BN={bn:3d} warps={nw} stages={ns}  ({err_msg[:80]})")
            unsafe.append({"bm": bm, "bn": bn, "nw": nw, "ns": ns, "err": err_msg})

    print()
    print(f"safe: {len(safe)} / {len(configs)}")
    if safe:
        ranked = sorted(safe, key=lambda c: c["ms"])
        print()
        print(f"top {min(10, len(ranked))} fastest safe configs:")
        for r in ranked[:10]:
            print(f"  BM={r['bm']:3d} BN={r['bn']:3d} warps={r['nw']} stages={r['ns']}  {r['ms']:7.4f} ms")

    return {"safe": safe, "unsafe": unsafe}


@app.local_entrypoint()
def main() -> None:
    result = probe_all.remote()
    safe = result["safe"]
    print()
    print("== summary ==")
    print(f"  configs safe:   {len(safe)}")
    print(f"  configs unsafe: {len(result['unsafe'])}")
    if safe:
        ranked = sorted(safe, key=lambda c: c["ms"])
        print(f"  fastest: BM={ranked[0]['bm']} BN={ranked[0]['bn']} "
              f"warps={ranked[0]['nw']} stages={ranked[0]['ns']}  {ranked[0]['ms']:.4f} ms")
