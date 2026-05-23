# Kernels

Design and performance notes for scree's varlen kernels — both the reference (slow but correct) impls and the fast Triton ones.

## Reference kernels

Located in [`src/scree/kernels/reference/`](../src/scree/kernels/reference/). Pure Python (NumPy / PyTorch / MLX), used as the ground truth in CI. **Not for production speed** — they iterate Python-level over sequences.

### `varlen_attention`

```python
from scree.kernels.reference import varlen_attention
out = varlen_attention(q, k, v, causal=False)
```

Computes per-sequence attention. For each sequence `i`:

```
scores_i  = q_i @ k_i^T / sqrt(head_dim)        # (heads, L_i, L_i)
if causal: scores_i[h, j, k] = -inf for k > j
attn_i    = softmax(scores_i, axis=-1)
out_i     = attn_i @ v_i                         # (heads, L_i, head_dim) -> (L_i, heads, head_dim)
```

All sequences are processed independently; there is no attention between sequences.

The implementation uses `numpy.einsum` (or `torch.einsum`, `mx.einsum`) for clarity. It loops over batch in Python.

### `varlen_layernorm`

```python
from scree.kernels.reference import varlen_layernorm
y = varlen_layernorm(arr, weight=W, bias=b, eps=1e-5)
```

LayerNorm is per-token — each row's mean and variance are computed over the last (feature) dim only, with no interaction with other rows. This means it's elementwise on the packed `values` buffer, no per-sequence loop needed.

### `varlen_rmsnorm`

```python
from scree.kernels.reference import varlen_rmsnorm
y = varlen_rmsnorm(arr, weight=W, eps=1e-6)
```

RMSNorm replaces LayerNorm in every modern open LLM (LLaMA, Mistral, Mixtral, DeepSeek, Qwen). Drops the mean-subtraction step from LayerNorm — `y = x / sqrt(mean(x²) + eps) * weight`. Same elementwise structure on the packed buffer.

### `varlen_softmax`

```python
from scree.kernels.reference import varlen_softmax
p = varlen_softmax(arr)
```

Softmax along the ragged dim, per-sequence. Unlike layernorm, softmax DOES need a per-sequence loop because the denominator is the sum within each sequence:

```
out_i = exp(arr_i - max(arr_i)) / sum(exp(arr_i - max(arr_i)))
```

If you softmaxed the full packed buffer along axis 0, you'd accidentally mix sequences. The kernel iterates explicitly to prevent this.

### Why these three

These are the operations that come up in every transformer block:

- Attention (the headline op)
- LayerNorm (pre-norm or post-norm at every block)
- Softmax (used in attention internally but also for routing, MoE)

The combination is enough to build a full transformer forward pass — see [`examples/02_no_pad_transformer.py`](../examples/02_no_pad_transformer.py).

Other ops likely to land in v0.2: `varlen_rope`, `scatter_add`/`gather` (the MoE routing primitives), `varlen_dropout`.

## Triton kernels

Located in [`src/scree/kernels/triton/`](../src/scree/kernels/triton/). CUDA-only — uses Triton's GPU code generation. The first call autotunes; subsequent calls hit the cached config.

### `varlen_attention_triton`

The headline GPU kernel.

```python
from scree.kernels.triton import varlen_attention_triton
out = varlen_attention_triton(q, k, v, cu_seqlens, causal=True)
```

Implements FlashAttention-2 style online-softmax forward, varlen variant.

### Algorithm

The kernel processes the output in tiles of `BLOCK_M` queries. For each Q-tile, it streams through K/V tiles of `BLOCK_N` and maintains a running softmax using the online-softmax recurrence:

```
For each Q tile (BLOCK_M rows of q):
    m_i = -inf                  # running max
    l_i = 0                     # running denominator
    acc = 0                     # accumulator (BLOCK_M, head_dim)

    For each K/V tile (BLOCK_N rows of k, v):
        scores = q @ k^T * scale                # (BLOCK_M, BLOCK_N)
        scores = mask_padding(scores, k_idx < seq_len)
        if causal: scores = mask_causal(scores, q_idx, k_idx)

        m_new = max(m_i, max(scores, axis=1))   # new running max
        alpha = exp(m_i - m_new)                # rescale old acc to new max
        p = exp(scores - m_new)                 # this tile's exponents
        l_new = alpha * l_i + sum(p, axis=1)

        acc = acc * alpha + p @ v               # accumulate weighted v's
        m_i, l_i = m_new, l_new

    out = acc / l_i                             # normalize
```

This is the standard FlashAttention-2 forward, with two differences for varlen:

1. **Per-sequence boundaries.** The launch grid is `(batch, q_block_idx, head)`. Each program reads `cu_seqlens[batch]` and `cu_seqlens[batch+1]` to find its sequence's start/end in the packed buffer. K/V tiles past the sequence end are masked out.
2. **Causal mask within-sequence.** When `causal=True`, the mask is `q_idx >= k_idx` where both indices are *local* to the sequence — a query at position 3 in sequence 2 cannot attend to position 5 in sequence 2, and cannot attend to anything in sequence 1.

### Tile shapes and autotuning

The kernel is decorated with `@triton.autotune` over a 24-config grid:

```
BLOCK_M    ∈ {64, 128}
BLOCK_N    ∈ {32, 64, 128}
num_warps  ∈ {4, 8}
num_stages ∈ {2, 3}
```

On the first call, Triton runs each config once, measures, picks the fastest, and caches the choice for the lifetime of the process. The autotune overhead is ~100-200ms; amortized across thousands of subsequent calls.

The `key=[]` on the autotuner means "tune once, use forever in this process." This is intentional — when head_dim changes, the kernel needs to be re-JIT'd anyway (HEAD_DIM is constexpr), so re-running autotune costs nothing extra.

### Performance

On NVIDIA H100 80GB SXM, fp16, causal, 16 sequences × log-normal lengths (mean ~760 tokens, max 2048), 16 heads × head_dim 64:

| Kernel | Time | Ratio |
| --- | --- | --- |
| FlashAttention-2 varlen | 0.166 ms | 1.00× |
| scree-Triton varlen     | 0.201 ms | 1.21× |

This is the first-attempt result from the unautotuned kernel measured at commit `12d7579`. With autotuning enabled (commit after this docs work), the ratio is expected to drop. Reproduce with:

```bash
modal run benchmarks/modal_bench.py
```

See [benchmarks.md](benchmarks.md) for methodology.

### Why FA-2 style, not FA-3 or FlexAttention

- **FA-3** uses Hopper-specific instructions (TMA, WGMMA). The kernel would need substantial rewriting and would lose portability to Ampere (A100). FA-2 style runs well on both.
- **FlexAttention** is PyTorch's "use any mask function" path. It composes well but requires PyTorch infrastructure — a goal for scree is to be portable beyond PyTorch.

The plan is to add FA-3 style and FlexAttention paths in v0.2 as optional fast paths; the FA-2 style kernel stays as the default.

### Why fp16/bf16 only

The current kernel uses `tl.dot` which requires the operand dtype to match. Supporting fp32 would require either an fp32-only kernel or up-casting at every accumulator boundary; neither is high priority because real transformer training is fp16/bf16.

fp32 reference impl exists in `scree.kernels.reference.varlen_attention`. If you need fp32 + GPU speed, file an issue.

### What the kernel does NOT do (v0.0)

- **No backward pass.** Forward only. Backward is the v0.1 → v0.2 work.
- **No GQA (grouped-query attention).** Q, K, V all have the same number of heads. GQA is a small change (different head strides for Q vs K/V) and will land in v0.2.
- **No sliding-window / sparsity patterns.** These can be implemented with custom mask functions but the v0.0 kernel hard-codes "full or causal."
- **No KV cache concat.** Inference-time KV-cache append requires a separate kernel path that takes the new tokens + old KV and concatenates. Planned for v0.2.

## Adding a new optimized kernel

The contract for adding a new kernel:

1. **Reference impl first** — add the operation to `src/scree/kernels/reference/<op>.py` for all backends (NumPy, PyTorch, MLX).
2. **Test against an obvious baseline** — in `tests/test_varlen_kernels.py`, compare your reference impl against a per-sequence padded baseline. The test should fail if you broke the math.
3. **Optimized impl** — add the Triton (or CUDA, or Metal) kernel in `src/scree/kernels/<backend>/<op>.py`.
4. **Cross-validation** — the optimized kernel must agree with the reference within declared FP tolerance, on a non-trivial input.
5. **Benchmark** — add a benchmark in `benchmarks/` that compares your kernel to the relevant baseline (e.g., a torch native op, FlashAttention, etc.).

The reference impl is always the source of truth. If the optimized impl disagrees with the reference, the optimized impl is wrong.

## Reading the Triton kernel

If you want to read [`src/scree/kernels/triton/varlen_attention.py`](../src/scree/kernels/triton/varlen_attention.py), the bird's-eye structure:

1. **`_autotune_configs()`** builds the search grid.
2. **`@triton.autotune(...)`** wraps the JIT kernel.
3. **`@triton.jit def _varlen_attn_fwd_kernel(...)`** is the device-side function. Read it top-to-bottom: load Q tile, init accumulators, loop over K/V tiles with online softmax, write output.
4. **`varlen_attention_triton(...)`** is the host-side launcher. It validates inputs, computes the grid shape, and invokes the kernel.

Triton kernels look like NumPy with `tl.` prefixes. The unusual primitives are:

- `tl.program_id(axis)` — which program in the launch grid am I?
- `tl.load(ptrs, mask=, other=)` — coalesced load with optional masking
- `tl.store(ptrs, value, mask=)` — coalesced store
- `tl.dot(a, b)` — matmul on Hopper/Ampere tensor cores

The rest is just NumPy. Don't be afraid of the file.

## References

- Tri Dao, FlashAttention-2 paper: <https://arxiv.org/abs/2307.08691>
- Triton tutorials, esp. "Fused Attention": <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- The FlashAttention reference Triton impl: <https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py>
