# API reference

Every public function and class in `scree`, with signatures and expected behavior.

The library is intentionally small: one data class, five core functions, six bridges, four reference kernels, one Triton kernel.

---

## `scree.Array`

```python
@dataclass(frozen=True)
class scree.Array:
    values: Any
    offsets: Any
    ragged_dim: int = 0
```

A packed values + offsets array with one variable-length dimension.

### Construction

`Array` is a `@dataclass(frozen=True)`. You can construct it directly:

```python
import numpy as np
values = np.zeros((10, 4), dtype=np.float32)
offsets = np.array([0, 3, 7, 10], dtype=np.int32)
arr = scree.Array(values=values, offsets=offsets, ragged_dim=0)
```

…but in practice you'll go through `scree.pack` or a `scree.from_*` constructor.

### Invariants enforced in `__post_init__`

| Invariant | Raised |
| --- | --- |
| `values.ndim >= 1` | `ValueError("values must be at least 1-D, ...")` |
| `0 <= ragged_dim < values.ndim` | `ValueError("ragged_dim={x} out of range ...")` |
| `offsets.ndim == 1` | `ValueError("offsets must be 1-D, ...")` |
| `len(offsets) >= 2` | `ValueError("offsets must have length >= 2")` |
| `int(offsets[0]) == 0` | `ValueError("offsets[0] must be 0, ...")` |
| `int(offsets[-1]) == values.shape[ragged_dim]` | `ValueError("offsets[-1] ({x}) must equal values.shape[ragged_dim={y}] ({z})")` |

Monotonicity of `offsets` is not checked. The caller is trusted.

### Properties

| Property | Type | Returns |
| --- | --- | --- |
| `batch_size` | `int` | `len(offsets) - 1` |
| `lengths` | array | `offsets[1:] - offsets[:-1]` — per-row lengths |
| `total_length` | `int` | `int(offsets[-1])` |
| `dtype` | dtype | `values.dtype` |
| `feature_shape` | `tuple` | shape of `values` with the ragged dim removed |

### Dunder methods

- `len(arr)` → `arr.batch_size`
- `repr(arr)` → `"scree.Array(batch_size=B, total_length=T, feature_shape=F, dtype=D)"`

---

## Construction and conversion

### `scree.pack(arrays, ragged_dim=0) -> Array`

Pack a list of arrays into one `scree.Array`.

- All arrays must share `dtype` and all non-ragged dims.
- The first array's backend determines the result backend (NumPy / PyTorch / MLX).
- Empty list raises `ValueError("Cannot pack an empty list")`.

```python
arr = scree.pack([torch.randn(3, 4), torch.randn(5, 4), torch.randn(2, 4)])
# arr.batch_size == 3, arr.total_length == 10
```

### `scree.unpack(arr) -> list`

Inverse of `pack`. Returns a list of `batch_size` arrays.

Returned slices are views into the original `values` where the backend supports it (NumPy and PyTorch always; MLX produces zero-copy lazy views).

```python
rows = scree.unpack(arr)
# len(rows) == arr.batch_size
# rows[i].shape[arr.ragged_dim] == int(arr.offsets[i+1] - arr.offsets[i])
```

### `scree.to_padded(arr, side="right", fill_value=0.0) -> (padded, mask)`

Materialize a `(batch, max_len, *feature_shape)` dense tensor + a `(batch, max_len)` bool mask.

| Parameter | Type | Meaning |
| --- | --- | --- |
| `side` | `"right"` or `"left"` | Where to place padding |
| `fill_value` | float | Padding value (default 0.0) |

Constraints (v0.1):

- Only `ragged_dim == 0` is supported.

Returns:

- `padded` — same dtype as `arr.values`, padded with `fill_value`
- `mask` — bool array, `True` for valid positions

### `scree.from_padded(padded, mask) -> Array`

Inverse of `to_padded` assuming **right-padding**. Reads `mask.sum(axis=1)` to get lengths.

```python
arr = scree.from_padded(padded, mask)
# arr.values is the concatenation of padded[i, :lengths[i]] for each i
```

### `scree.from_cu_seqlens(values, cu_seqlens) -> Array`

Construct from FlashAttention's `cu_seqlens` convention. `cu_seqlens` IS the offsets format — this is zero-copy:

```python
arr = scree.from_cu_seqlens(q, cu_seqlens_q)
assert arr.values is q
assert arr.offsets is cu_seqlens_q
```

---

## `scree.bridges` — migration paths

### `scree.bridges.to_torch_nested(arr) -> torch.Tensor`

Convert to a `torch.nested.nested_tensor(layout=torch.jagged)`. Materializes per-row slices and hands them to `torch.nested.nested_tensor` — may copy internally depending on torch version.

Constraints:

- `arr.ragged_dim` must be 0.
- `arr.values` must be a torch tensor (raises `ImportError` if torch is unavailable).

### `scree.bridges.from_torch_nested(nt) -> Array`

Convert a torch jagged NestedTensor back to a `scree.Array`. Uses `nt.values()` and `nt.offsets()` on modern torch versions; falls back to `unbind() + cat` for older versions.

Raises `TypeError` if `nt` is not a NestedTensor.

### `scree.bridges.to_hf_padded(arr) -> (hidden_states, attention_mask)`

Convert to the HuggingFace Transformers convention.

Returns:

- `hidden_states` — right-padded dense tensor
- `attention_mask` — `int64`, `1` for valid positions, `0` for padding (HF convention)

### `scree.bridges.from_hf_padded(hidden_states, attention_mask) -> Array`

Inverse of `to_hf_padded`. Accepts any integer or bool mask; converts to bool internally.

### `scree.bridges.to_torch(arr) -> Array`

Re-export a `scree.Array` with its values/offsets as torch tensors. Zero-copy on CPU via `torch.from_numpy`; DLPack on GPU.

If `arr` is already a torch-backed `scree.Array`, returns it unchanged.

### `scree.bridges.to_numpy(arr) -> Array`

Re-export with values/offsets as numpy arrays. Zero-copy from CPU torch tensors; copies from GPU torch tensors to host.

If `arr` is already numpy-backed, returns it unchanged.

---

## `scree.kernels.reference` — slow but correct

These are pure Python (or PyTorch/MLX) reference implementations used as ground truth in CI tests of the Triton kernels. They are correct but not fast. For production, use `scree.kernels.triton` on CUDA.

### `varlen_attention(q, k, v, causal=False) -> Array`

Variable-length self-attention. Each sequence attends only to itself — no cross-sequence attention.

| Parameter | Type | Shape |
| --- | --- | --- |
| `q`, `k`, `v` | `Array` | `(total_len, n_heads, head_dim)` each, with identical offsets |
| `causal` | bool | Apply lower-triangular mask within each sequence |

Returns an `Array` with the same offsets as `q`.

Raises `ValueError` if `q`, `k`, `v` don't have identical offsets.

### `varlen_layernorm(arr, weight=None, bias=None, eps=1e-5) -> Array`

LayerNorm over the last dim of `arr.values`.

| Parameter | Type | Meaning |
| --- | --- | --- |
| `weight` | array or `None` | Scale parameter of shape `(feature_dim,)` |
| `bias` | array or `None` | Shift parameter of shape `(feature_dim,)` |
| `eps` | float | Numerical stability epsilon |

LayerNorm is per-token (no cross-row interaction) so it's elementwise on the packed buffer.

### `varlen_rmsnorm(arr, weight=None, eps=1e-6) -> Array`

RMSNorm (Zhang & Sennrich, 2019) over the last dim — the norm used by
LLaMA, Mistral, Mixtral, DeepSeek, Qwen, and most post-2023 open transformers.
Differs from LayerNorm by dropping mean subtraction; `y = x / rms(x) * weight`.

| Parameter | Type | Meaning |
| --- | --- | --- |
| `weight` | array or `None` | Scale parameter of shape `(feature_dim,)` |
| `eps` | float | Numerical stability epsilon (typical 1e-6 for LLaMA-family) |

Like LayerNorm, RMSNorm is per-token — elementwise on the packed buffer.

### `varlen_softmax(arr) -> Array`

Softmax along the ragged dim, per-sequence. Each row is softmaxed independently.

Constraint (v0.1): `arr.ragged_dim == 0`.

---

## `scree.kernels.triton` — fast (CUDA + Triton)

### `varlen_attention_triton_autograd(q, k, v, cu_seqlens, causal=False) -> torch.Tensor`

Autograd-aware variant of `varlen_attention_triton`. Both forward and backward are GPU-native Triton kernels (FA-2 style). Use this when you need gradients to flow through `q`, `k`, `v`.

Returns a `torch.Tensor` of the same shape and dtype as `q`. On `.backward()`, dispatches to `_varlen_attn_bwd_preprocess_kernel`, `_varlen_attn_bwd_dkv_kernel`, and `_varlen_attn_bwd_dq_kernel` to compute `dq`, `dk`, `dv`.

H100 measurement (12k tokens, fp16, causal): full training step at **1.61× of FA-2 varlen**.

### `varlen_attention_triton(q, k, v, cu_seqlens, causal=False, return_lse=False) -> torch.Tensor`

GPU implementation of varlen self-attention forward using a FlashAttention-2 style online-softmax recurrence with autotuned block sizes.

| Parameter | Type | Shape |
| --- | --- | --- |
| `q`, `k`, `v` | `torch.Tensor` | `(total_tokens, n_heads, head_dim)`, fp16 or bf16, CUDA |
| `cu_seqlens` | `torch.Tensor` | `(batch + 1,)`, int32 — IS the `scree.Array.offsets` format |
| `causal` | bool | Apply lower-triangular causal mask within each sequence |

Block sizes (`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`) are chosen by `triton.autotune` on first call and cached for the process lifetime.

Returns a `torch.Tensor` of the same shape and dtype as `q`.

Raises `RuntimeError` if Triton is unavailable. Raises `AssertionError` on non-CUDA tensors or non-fp16/bf16 dtypes.

Importing `scree.kernels.triton` is safe on non-CUDA platforms: `TRITON_AVAILABLE` is `False` and no kernel symbols are exported. Calling the kernel without Triton raises an informative error.

---

## `scree.__version__`

The version string. v0.0.x is pre-alpha; v0.1 introduces a stable API.

---

## What's NOT in the public API

These names exist in `scree._core` but are not exported:

- `_is_torch(x)`, `_is_mlx(x)` — backend dispatch predicates. Useful inside `scree.*` but considered private.

Anything else with a leading underscore in any module is private and subject to change without notice.
