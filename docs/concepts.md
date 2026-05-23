# Concepts

The mental model behind `scree.Array`. Read this once and the API stops surprising you.

## The data structure

A `scree.Array` is three things:

```
values     — a dense N-dimensional array storing all rows concatenated
offsets    — a 1-D int array marking row boundaries
ragged_dim — which dim of `values` is variable-length (default 0)
```

If you have three sequences of lengths `[4, 2, 5]`, each with a feature dim of 8:

```
values:   shape (11, 8)    # 4 + 2 + 5 = 11 along the ragged dim
offsets:  [0, 4, 6, 11]    # length B+1; offsets[i+1] - offsets[i] = length of row i
ragged_dim: 0
```

That's it. Everything in the library is built on top of this triple.

## Why this representation

There are five incompatible ways the field stores variable-length data today:

| Approach | Pros | Cons |
| --- | --- | --- |
| **Pad + mask** (HF Transformers) | dense math is easy | wastes memory and FLOPs on padding |
| **List of tensors** | simple to construct | every op iterates Python; no GPU batching |
| **`torch.nested`** (PyTorch) | typed, fast on PyTorch | PyTorch-only, incomplete since 2021 |
| **`RaggedTensor`** (TensorFlow) | typed | TF-only, declining ecosystem |
| **`cu_seqlens`** (FlashAttention) | what kernels actually want | a convention, not a typed primitive |

The packed `values + offsets` layout is what every fast kernel wants internally. FlashAttention's `cu_seqlens` is exactly this. vLLM's "continuous batching" is exactly this. SGLang's batched layout is exactly this. `scree.Array` is what they all converge on, exposed as a single neutral type.

## The invariants

A `scree.Array` enforces, at construction time:

1. `values.ndim >= 1`
2. `0 <= ragged_dim < values.ndim`
3. `offsets.ndim == 1`
4. `len(offsets) >= 2` — you can't have zero sequences
5. `offsets[0] == 0`
6. `offsets[-1] == values.shape[ragged_dim]`

Monotonicity (`offsets[i+1] >= offsets[i]`) is not enforced for performance — it's assumed. Constructing with non-monotonic offsets is undefined behavior. Don't.

## The relationship to FlashAttention

FlashAttention's `flash_attn_varlen_func` takes:

```
q, k, v       — packed (total_tokens, n_heads, head_dim)
cu_seqlens_q  — length B+1, int32
cu_seqlens_k  — length B+1, int32
```

This is `scree.Array` with `n_heads × head_dim` as the feature_shape:

```python
arr = scree.from_cu_seqlens(values, cu_seqlens)   # zero-copy
```

`scree.from_cu_seqlens` is literally a no-op aside from constructing the dataclass — `offsets` *is* `cu_seqlens`, the same int32 buffer.

## The relationship to HuggingFace's `attention_mask`

HF Transformers passes variable-length data as `(hidden_states, attention_mask)`:

```
hidden_states:   shape (batch, seq_len, *features)        # right-padded
attention_mask:  shape (batch, seq_len), int 1/0          # 1 = real token, 0 = pad
```

Converting to scree:

```python
arr = scree.bridges.from_hf_padded(hidden_states, attention_mask)
```

This is **not** zero-copy — the padding tokens are dropped and the real ones are repacked into a flat buffer. But the result is much smaller in memory (often 70–85% smaller on realistic LLM length distributions; see [benchmarks.md](benchmarks.md)).

## When NOT to use scree

scree is *purely* a primitive for one specific shape of data: variable-length sequences along one axis with otherwise dense dimensions. Use something else when:

- **You have only one sequence at a time.** Just use a regular tensor. scree adds bookkeeping that helps only when you have multiple sequences in the same buffer.
- **Your variability is across multiple axes** (e.g., a list of 2-D matrices of varying both rows and columns). scree supports exactly one ragged dim per `Array`. For two-axis ragged data, you'd need a sparse tensor primitive, not scree.
- **You need a graph data structure.** Use PyG / DGL — they're optimized for adjacency, not just length.
- **You're already happy with `torch.nested`.** If you're PyTorch-only, your kernels accept `torch.nested`, and you don't need cross-framework, stick with what works.

## Dispatch and backend selection

`scree.Array` is backend-agnostic: `values` and `offsets` can be NumPy arrays, PyTorch tensors, or MLX arrays. The functions in `scree.*` detect the backend at runtime and dispatch.

```python
type(arr.values).__module__   # 'numpy' | 'torch' | 'mlx.core'
```

This is implemented with two helper predicates inside [`scree._core`](../src/scree/_core.py):

```python
def _is_torch(x): return type(x).__module__.startswith("torch")
def _is_mlx(x):   return type(x).__module__.startswith("mlx")
```

Anything not torch or mlx falls into the NumPy code path. This is intentionally simple — no plugin registry, no protocol — and is sufficient for v0.x. When a fourth backend is added (likely JAX), the predicate joins the chain.

## Why backends instead of a single Array API

scree could have been built on top of the [Python Array API](https://data-apis.org/array-api/) via [`array-api-compat`](https://github.com/data-apis/array-api-compat) and avoided per-backend dispatch entirely. That was considered and rejected for v0.x because:

1. **Some operations are not in the Array API.** In-place mutation (`padded[i, :length] = row`) is the obvious one — JAX doesn't allow it; NumPy and PyTorch do.
2. **Backends have idiomatic perf paths.** `torch.cat` vs `torch.nested.nested_tensor`, `mx.softmax` vs hand-rolled, etc. A unified namespace would hide these.
3. **The dispatch is small.** With three backends and five operations, the duplicated branches are about 200 lines. The cost of indirection through array-api-compat would exceed the cost of duplication.

When/if the Array API spec covers in-place mutation and the major backends adopt it cleanly, scree can switch.

## Reading further

- [api.md](api.md) — exact signatures and behavior for every function
- [architecture.md](architecture.md) — how the package is laid out internally
- [kernels.md](kernels.md) — how the reference and Triton kernels work
