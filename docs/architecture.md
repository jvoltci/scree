# Architecture

How the package is laid out internally, what each module is responsible for, and why the design choices were made.

This document is for people who want to contribute to scree or understand the codebase deeply. If you just want to use the library, start with [getting-started.md](getting-started.md) and [concepts.md](concepts.md).

## Repository layout

```
scree/
├── pyproject.toml                 # package metadata, optional deps
├── README.md                      # one-screen pitch, status table
├── LICENSE                        # Apache-2.0
├── CONTRIBUTING.md                # how to propose changes
├── CHANGELOG.md                   # version history
├── .github/workflows/ci.yml       # CI: tests on Python 3.10/3.11/3.12, ubuntu + macos
├── docs/                          # this directory
├── src/scree/                     # the package source (src-layout)
│   ├── __init__.py                # public API surface
│   ├── _core.py                   # Array dataclass + 5 core ops + backend predicates
│   ├── bridges/                   # migration paths to existing tools
│   │   ├── __init__.py
│   │   ├── _torch_nested.py
│   │   ├── _hf_padded.py
│   │   └── _dlpack.py
│   └── kernels/
│       ├── __init__.py
│       ├── reference/             # slow Python impls used as CI ground truth
│       │   ├── varlen_attention.py
│       │   ├── varlen_layernorm.py
│       │   └── varlen_softmax.py
│       └── triton/                # fast GPU kernels, import-gated for non-CUDA
│           ├── __init__.py        # gates the import; sets TRITON_AVAILABLE
│           └── varlen_attention.py
├── tests/                         # pytest suite, 31 tests across 3 backends
│   ├── test_core.py
│   ├── test_bridges.py
│   ├── test_varlen_kernels.py
│   └── test_mlx.py
├── benchmarks/
│   ├── README.md
│   ├── bench_memory.py            # CPU memory benchmark vs HF padded
│   └── modal_bench.py             # H100 benchmark of Triton kernel vs FA-2
└── examples/
    ├── 01_quickstart.py           # 6-line demo
    └── 02_no_pad_transformer.py   # full transformer block on scree primitives
```

## Module responsibilities

### `scree._core`

The center of the library. Defines:

- The `Array` dataclass and its invariants
- The five core operations: `pack`, `unpack`, `to_padded`, `from_padded`, `from_cu_seqlens`
- Two backend predicates: `_is_torch(x)`, `_is_mlx(x)`

Everything else in the package imports from `_core`. There is exactly one source of truth for the data structure.

### `scree.bridges`

A flat namespace of migration helpers. Each file in `bridges/` handles one external convention:

- `_torch_nested.py` — round-trip with PyTorch's jagged NestedTensor
- `_hf_padded.py` — round-trip with the HuggingFace `(hidden_states, attention_mask)` pair
- `_dlpack.py` — re-export `values`/`offsets` between NumPy and PyTorch via DLPack

The bridges are deliberately *not* in `_core` because:

1. They import heavy optional deps (torch in particular) — keeping them in a separate module means importing `scree.Array` doesn't drag in torch.
2. They are one-directional helpers, not part of the primitive abstraction.

### `scree.kernels.reference`

Pure-Python (NumPy / PyTorch / MLX) implementations of the three reference kernels:

- `varlen_attention`
- `varlen_layernorm`
- `varlen_softmax`

These are intentionally slow and obviously correct. They serve as ground truth in [`tests/test_varlen_kernels.py`](../tests/test_varlen_kernels.py) and [`tests/test_mlx.py`](../tests/test_mlx.py) — every optimized kernel must produce numerically equivalent output.

The dispatch chain in each reference kernel is consistent:

```python
if _is_mlx(arr.values):
    # MLX path
elif _is_torch(arr.values):
    # PyTorch path
else:
    # NumPy path
```

MLX comes first because its in-place limitations require alternative code paths; checking it first lets the more permissive branches act as defaults.

### `scree.kernels.triton`

The fast GPU kernels.

The package's `__init__.py` is **import-gated**: it tries to `import triton` and sets `TRITON_AVAILABLE = True/False` accordingly. On platforms without Triton (macOS, CPU-only Linux, etc.) the import is silent and no kernel names are exported. This lets `import scree` and `import scree.kernels.triton` work everywhere.

Currently ships:

- `varlen_attention_triton` — FA-2 style varlen self-attention forward

The kernel uses `@triton.autotune` with a 24-config grid over `(BLOCK_M, BLOCK_N, num_warps, num_stages)`; the first call for a given workload selects the best config and caches it.

## Design choices and rationale

### Why a dataclass, not a class

`scree.Array` is `@dataclass(frozen=True)`. The frozenness gives:

- Hashability — the same `Array` value hashes the same way (useful for memoization)
- Safety — code that receives an `Array` can't accidentally mutate offsets and break invariants

The dataclass shape (three fields, no methods) signals that the type is a value object. All operations live in the surrounding module as free functions.

### Why backend dispatch instead of a unified namespace

Considered using [`array-api-compat`](https://github.com/data-apis/array-api-compat) to write the operations once in the Array API. Rejected for v0.x:

- Some operations (in-place mutation in `to_padded` / `from_padded`) don't have a clean Array API form because JAX disallows them.
- Per-backend idiomatic paths (`torch.cat`, `mx.softmax`) are faster than the Array API alternatives.
- The dispatch is small (200 lines) and easy to maintain.

We can revisit when the Array API spec covers in-place mutation or when we add a fourth backend.

### Why import-gate the Triton module

Triton requires a CUDA-capable GPU and PTX. On macOS/Apple Silicon/CPU-only servers, `import triton` raises `ImportError`. If the scree package eagerly imported triton, the library would be unimportable on the developer's M5 — including just running the test suite.

The gate (`try: import triton`) lets the same package serve both groups. `TRITON_AVAILABLE` is the contract.

### Why `cu_seqlens` is just `offsets`

FlashAttention's `cu_seqlens` is `[0, L0, L0+L1, ...]`. scree's `offsets` is exactly that. The choice to use the same convention is deliberate: `scree.from_cu_seqlens(values, cu_seqlens)` is effectively a no-op aside from the dataclass construction.

This means scree can plug into the FlashAttention / vLLM / SGLang kernel ecosystem with zero conversion overhead. Adopting a different convention (e.g., `lengths` instead of `offsets`) would have required either round-trip conversion at every kernel boundary or a fork of the kernel API.

### Why one ragged dim, not multiple

Real workloads have variable length along *one* axis (the sequence axis). Other axes (head, feature, batch position) are dense. Supporting multiple ragged dims would have made the data structure substantially more complex (a tree of offsets instead of a single 1-D array) without serving real use cases.

Multi-ragged use cases (e.g., graphs where both nodes and edges vary) are served by sparse tensor primitives (PyG, DGL), not by scree.

### Why no autograd integration

scree.Array is not a `torch.Tensor` subclass and does not register with PyTorch's autograd. The `values` field IS a torch tensor that participates in autograd normally; the wrapper is just a typed view.

This is intentional. Tying scree to PyTorch's autograd would:

- Make the type PyTorch-specific, breaking the cross-framework story
- Force users to choose between scree and other tensor-wrapping types (e.g., named tensors)

The autograd "lives in" `values`. Operations that need to propagate gradients (varlen_attention, layernorm) construct a new `scree.Array` whose `values` is the autograd-tracked output. This works cleanly because `scree.Array` is just a wrapper.

## Test infrastructure

The test suite uses pytest. Three test files cover the three concern areas:

- `tests/test_core.py` — `Array` invariants, `pack`/`unpack` round-trip, `to_padded`/`from_padded` round-trip, NumPy and PyTorch paths
- `tests/test_varlen_kernels.py` — reference kernel correctness on NumPy
- `tests/test_bridges.py` — every bridge round-trip on NumPy + PyTorch
- `tests/test_mlx.py` — MLX backend tests including cross-backend numerical agreement (`atol=5e-3` to account for MLX's tensor-core mixed-precision matmul)

Total: 31 tests as of v0.0.x. CI runs them on Python 3.10/3.11/3.12 (Ubuntu) and 3.11 (macOS), plus runs both examples and the memory benchmark to catch bitrot.

The Triton kernel is not tested in CI (CI doesn't have a CUDA GPU). Its correctness is verified by `benchmarks/modal_bench.py`, which runs on an H100 via Modal and asserts max abs diff < 5e-3 against FlashAttention-2. This runs on demand, not on every push.

## Performance characteristics

Memory: scree's footprint is exactly `values.nbytes + offsets.nbytes`. For a batch of B sequences with mean length L and feature dim D in fp32:

- scree: `4 * B * L * D + 4 * (B + 1)` bytes
- HF padded: `4 * B * Lmax * D + 4 * B * Lmax` bytes (mask is int32 in HF; bool in scree.to_padded)

The ratio depends on the variance of lengths. For log-normal distributions with σ=0.6 the scree representation is ~70% smaller; with σ=1.2 it's ~85% smaller. See [benchmarks.md](benchmarks.md).

Throughput: the reference kernels loop in Python and are O(B × kernel_op_cost). They're correctness-only — production use needs the Triton path on CUDA. On H100, scree's autotuned Triton `varlen_attention` is within 1.0-1.21x of FlashAttention-2 varlen (first attempt, no compile-time specialization).

## Adding a new backend

To add JAX (the most likely next backend):

1. Add `_is_jax(x)` to `scree._core`.
2. Add JAX branches to `pack`, `unpack`, `to_padded`, `from_padded` — use `jnp.concatenate`, `jnp.cumsum`, and mutation-free construction (`jnp.stack(rows)`) the way MLX does.
3. Add JAX branches to each reference kernel.
4. Add `tests/test_jax.py` mirroring `tests/test_mlx.py`.
5. Add `jax` extra to `pyproject.toml`.
6. Update README's status table.

The MLX integration (commit `4876a21`) is the template — it added 200 lines across 7 files and one new test file. JAX should be similar.

## Adding a new kernel

A new optimized kernel lives in `src/scree/kernels/triton/` (or `cuda/`, `metal/`, etc. as the project grows). Requirements:

1. A pure-Python or PyTorch reference implementation in `src/scree/kernels/reference/`.
2. A test in `tests/test_varlen_kernels.py` that verifies the reference against an obvious correctness baseline (e.g., padded computation).
3. The Triton (or other backend) kernel matches the reference within declared FP tolerance.
4. A benchmark in `benchmarks/` that compares to the relevant external baseline (e.g., FlashAttention).

The reference impl is the contract. Optimized kernels are validated against it, not against each other.

## Versioning

scree follows semver:

- **v0.0.x** — pre-alpha, the API may break between any two commits
- **v0.1.0** — first beta release; the public API in `scree.*` becomes stable
- **v1.0.0** — first stable release; backwards-incompatible changes require a major bump

The v0.0 → v0.1 milestone gate is in [`benchmarks/README.md`](../benchmarks/README.md): the Triton kernel must hit ≤1.2× FA-2 with PASS correctness on H100. As of commit `12d7579` this is met (1.21×, PASS).

## Where decisions get made

- **Public API surface** — `src/scree/__init__.py` is the contract. Anything not re-exported there is private.
- **Behavior changes** — should be discussed in a GitHub Discussion before being implemented; PRs that change behavior without prior discussion will be asked to retroactively open one.
- **New backends or kernels** — a short GitHub Discussion thread (1-2 paragraphs) is enough — the maintainers want to encourage these.

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for the contribution flow.
