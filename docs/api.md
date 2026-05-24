# API reference

Auto-generated from the docstrings in `src/scree/`. If you find a discrepancy between
this page and the code, the code is right — please file an issue.

The library is intentionally small: one data class, five core operations, six bridges,
four reference kernels, three GPU Triton kernels.

---

## Core type and operations — `scree.*`

### `scree.Array`

::: scree.Array
    options:
      show_root_heading: false
      show_source: true
      members:
        - __post_init__
        - batch_size
        - lengths
        - total_length
        - dtype
        - feature_shape

### `scree.pack`

::: scree.pack
    options:
      show_root_heading: false

### `scree.unpack`

::: scree.unpack
    options:
      show_root_heading: false

### `scree.to_padded`

::: scree.to_padded
    options:
      show_root_heading: false

### `scree.from_padded`

::: scree.from_padded
    options:
      show_root_heading: false

### `scree.from_cu_seqlens`

::: scree.from_cu_seqlens
    options:
      show_root_heading: false

---

## Bridges — `scree.bridges`

Migration helpers between scree and existing ecosystem objects. Each bridge is
zero-copy where the underlying memory layout allows.

### `to_torch_nested` / `from_torch_nested`

::: scree.bridges.to_torch_nested
    options:
      show_root_heading: false

::: scree.bridges.from_torch_nested
    options:
      show_root_heading: false

### `to_hf_padded` / `from_hf_padded`

::: scree.bridges.to_hf_padded
    options:
      show_root_heading: false

::: scree.bridges.from_hf_padded
    options:
      show_root_heading: false

### `to_torch` / `to_numpy`

::: scree.bridges.to_torch
    options:
      show_root_heading: false

::: scree.bridges.to_numpy
    options:
      show_root_heading: false

---

## Reference kernels — `scree.kernels.reference`

Pure-Python (or PyTorch / MLX / JAX) implementations of the four varlen kernels. Used
as ground truth in CI tests of the optimized Triton kernels. **Not for production
speed** — they iterate Python-level over sequences.

### `varlen_attention`

::: scree.kernels.reference.varlen_attention
    options:
      show_root_heading: false

### `varlen_layernorm`

::: scree.kernels.reference.varlen_layernorm
    options:
      show_root_heading: false

### `varlen_rmsnorm`

::: scree.kernels.reference.varlen_rmsnorm
    options:
      show_root_heading: false

### `varlen_softmax`

::: scree.kernels.reference.varlen_softmax
    options:
      show_root_heading: false

---

## Triton kernels — `scree.kernels.triton`

CUDA-only. Importing `scree.kernels.triton` is safe on non-CUDA platforms
(`TRITON_AVAILABLE` is `False` and no kernel symbols are exported), but calling the
kernels without CUDA raises an informative error.

### `varlen_attention_triton`

The forward kernel — 1.30× of FA-2 on H100 for the headline workload.

### `varlen_attention_triton_autograd`

Autograd-aware wrapper. Forward + backward both run on Triton kernels (FA-2 style:
preprocess + dKV + dQ). Use this when you need gradients to flow through `q`, `k`,
`v`. Full training step at 1.61× of FA-2.

### `varlen_rmsnorm_triton`

13.97× speedup vs PyTorch reference on H100 (no native RMSNorm in PyTorch).

### `varlen_layernorm_triton`

1.31× speedup vs `torch.nn.functional.layer_norm` on H100.

---

## What's NOT in the public API

Names with a leading underscore in any module are private and subject to change
without notice. In particular:

- `scree._core._is_torch`, `_is_mlx`, `_is_jax` — backend dispatch predicates
- `scree.kernels.triton._varlen_attn_fwd_kernel` — the raw Triton kernel
- `scree.kernels.triton._varlen_attn_bwd_*_kernel` — raw backward kernels
- `scree.kernels.triton._backward.varlen_attention_triton_backward` — the host-side
  backward orchestrator (used by the autograd wrapper)
