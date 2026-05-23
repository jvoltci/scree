# Bridges & migration cookbook

Recipes for moving an existing project to scree without a rewrite.

The package's `scree.bridges` module turns this into one-line conversions in both directions. Pick the recipe that matches your current code.

## Coming from HuggingFace Transformers

Existing code:

```python
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
hidden_states = outputs.last_hidden_state    # (B, S, D)
```

Adopt scree at the boundary:

```python
import scree.bridges as bridges

# At the boundary where you receive HF outputs:
arr = bridges.from_hf_padded(hidden_states, attention_mask)

# arr.values is shape (total_real_tokens, D) — no padding wasted
# arr.offsets gives row boundaries
# arr.batch_size, arr.total_length etc. are available
```

When you need to call back into HF or a model that expects padded:

```python
hidden_states, attention_mask = bridges.to_hf_padded(arr)
outputs = next_model(inputs_embeds=hidden_states, attention_mask=attention_mask)
```

**Memory savings.** Realistic LLM batches (log-normal length distributions) save **70–85% memory** vs HF padded. See [benchmarks.md](benchmarks.md). Run it yourself:

```bash
python benchmarks/bench_memory.py
```

**HF mask convention.** HF uses `int64` masks with `1` for real tokens and `0` for padding. scree's bridge produces exactly this:

```python
_, mask = bridges.to_hf_padded(arr)
assert mask.dtype == torch.int64
assert ((mask == 0) | (mask == 1)).all()
```

## Coming from FlashAttention / vLLM `cu_seqlens`

If you already pack your data and pass `cu_seqlens` to FlashAttention or vLLM:

```python
# Existing code:
out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seq, max_seq, causal=True)
```

The `cu_seqlens` IS the scree offsets format. Conversion is zero-copy:

```python
arr_q = scree.from_cu_seqlens(q, cu_seqlens)
arr_k = scree.from_cu_seqlens(k, cu_seqlens)
arr_v = scree.from_cu_seqlens(v, cu_seqlens)

assert arr_q.values is q
assert arr_q.offsets is cu_seqlens
```

Now you can pass the `scree.Array` around your codebase as a typed primitive, then unpack at the kernel call:

```python
out = flash_attn_varlen_func(
    arr_q.values, arr_k.values, arr_v.values,
    arr_q.offsets, arr_k.offsets,
    int(arr_q.lengths.max()), int(arr_k.lengths.max()),
    causal=True,
)
```

Or use the bundled Triton kernel, which takes a `scree.Array`-flavored API directly:

```python
from scree.kernels.triton import varlen_attention_triton
out = varlen_attention_triton(arr_q.values, arr_k.values, arr_v.values, arr_q.offsets, causal=True)
```

## Coming from `torch.nested`

```python
import torch
nt = torch.nested.nested_tensor([...], layout=torch.jagged)
```

Convert:

```python
arr = scree.bridges.from_torch_nested(nt)
# arr.values is nt.values(); arr.offsets is nt.offsets() — zero-copy on modern torch
```

Going back:

```python
nt = scree.bridges.to_torch_nested(arr)
```

**When to do this conversion.** `torch.nested` works fine if you're PyTorch-only and your kernels accept it. The reason to move to `scree` is when you need to:

- Cross frameworks (NumPy / MLX / future JAX)
- Use kernels (FlashAttention, scree-Triton) that want `cu_seqlens`
- Share a batch between trainer and inference engine

## Coming from a list-of-tensors representation

```python
seqs = [embeddings_for_doc(doc) for doc in batch]   # list of variable-length tensors
```

Just pack:

```python
arr = scree.pack(seqs)
```

scree.pack works on numpy, torch, or mlx arrays. All elements must share dtype and the non-ragged dims.

## Coming from an ad-hoc padded + mask convention

You have `(padded, valid_mask)` where `valid_mask` is bool or int:

```python
arr = scree.from_padded(padded, valid_mask)
```

scree assumes right-padding. If you're left-padded, flip first:

```python
# If left-padded:
arr = scree.pack([padded[i, -lengths[i]:] for i in range(B)])
```

## The two-direction principle

Every bridge is symmetric:

| From | To |
| --- | --- |
| `from_hf_padded` | `to_hf_padded` |
| `from_torch_nested` | `to_torch_nested` |
| `from_cu_seqlens` | `arr.values, arr.offsets` (read directly) |
| `from_padded` | `to_padded` |

Round-trip identity holds (within fp tolerance for non-trivial values):

```python
arr2 = bridges.from_hf_padded(*bridges.to_hf_padded(arr))
assert (arr.values == arr2.values).all()
assert (arr.offsets == arr2.offsets).all()
```

This is also covered by the CI test suite in [`tests/test_bridges.py`](../tests/test_bridges.py).

## Cross-framework conversion

Move a `scree.Array` between NumPy and PyTorch (zero-copy on CPU, DLPack on GPU):

```python
arr_np = scree.pack([np.random.randn(n, 4).astype(np.float32) for n in [3, 5, 2]])
arr_t = bridges.to_torch(arr_np)            # torch tensors
arr_np2 = bridges.to_numpy(arr_t)           # back to numpy
```

For MLX, just pack with `mlx.core.array` inputs from the start; the result is MLX-backed and uses Apple Silicon's GPU via Metal.

```python
import mlx.core as mx
arr_mx = scree.pack([mx.random.normal((n, 4)) for n in [3, 5, 2]])
```

For NumPy ↔ MLX or PyTorch ↔ MLX, MLX exposes `mx.array(...)` and `np.array(mx_array)` — use those at the boundary, then re-pack if you need a scree.Array.
