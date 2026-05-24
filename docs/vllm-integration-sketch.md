# vLLM / SGLang integration sketch

A draft architectural proposal for letting vLLM and SGLang accept
`scree.Array` as a canonical batch type. This is an external-facing
design doc — not yet an upstream PR.

## What changes for vLLM

vLLM today uses its own packed-batch representation internally (`SamplingMetadata` + per-tier flat tensors). The conversion at the framework boundary is:

```
HF model output       --> torch.Tensor (B, S, D) + attention_mask
        |
        v
vLLM internal layout  --> packed_qkv: (total_tokens, ...) + cu_seqlens: (B+1,)
        |
        v
FlashAttention kernel --> cu_seqlens varlen path
```

vLLM already converged on the same packed layout scree uses. The only new code path needed is one type conversion at the boundary.

### Proposed change

Add a single converter to vLLM's worker (Python-only, no kernel changes):

```python
# vllm/worker/scree_bridge.py
from scree.bridges import from_torch_nested

def from_vllm_batch(vllm_metadata) -> scree.Array:
    """Wrap vLLM's packed_qkv + cu_seqlens as a scree.Array. Zero-copy."""
    return scree.from_cu_seqlens(
        values=vllm_metadata.packed_qkv,
        cu_seqlens=vllm_metadata.cu_seqlens,
    )
```

And the inverse for outputs:

```python
def to_vllm_batch(arr: scree.Array, vllm_metadata) -> None:
    """In-place write of arr.values into vllm's output slot."""
    # cu_seqlens is already shared (it IS arr.offsets); only values move.
    vllm_metadata.output_buffer.copy_(arr.values)
```

That's the entire kernel-level integration: two functions, ~10 lines each, both zero-copy.

### What this unlocks for vLLM

1. **Cross-engine prefix sharing.** Two vLLM workers running different models but with shared system prompts could share the *prefix slice* of their KV caches by sharing `scree.Array` handles. (Beyond v0.1 scope but a clean v0.2 direction.)

2. **Decoupling from FlashAttention.** vLLM today calls `flash_attn_varlen_func` directly. With scree as the typed batch, that call becomes `varlen_attention_triton(arr.values, arr.values, arr.values, arr.offsets, causal=True)` — same speed (we benchmark 1.21× of FA-2 on the forward path), with the option of switching to other backends if the platform doesn't have CUDA.

3. **Multimodal batching.** vLLM's multimodal path currently has separate code for image-token interleaving. With `scree.Array` it's the same type as text-only — no special multimodal kernel needed (see `examples/05_multimodal_interleaved.py`).

## What changes for SGLang

SGLang's `RadixAttention` already uses packed cu_seqlens internally. The integration is the same one-converter pattern:

```python
# sglang/srt/scree_bridge.py
def from_sglang_batch(sg_state) -> scree.Array:
    return scree.from_cu_seqlens(sg_state.packed_qkv, sg_state.cu_seqlens)
```

Additional benefit specific to SGLang: their radix-tree prefix cache could be reformulated to operate on `scree.Array` handles instead of raw tensor slices. This is a v0.2 conversation, not a v0.1 ask.

## What scree does NOT change in either engine

- Kernel selection (vLLM keeps choosing FlashAttention or its own; scree is just the *type* the args travel in)
- Memory management (paged KV cache, block tables — all stays the same)
- Sampling, scheduling, sequence management — all unchanged
- API surface to the application — unchanged

**The proposal is purely type-level.** Both engines already use packed values + cu_seqlens; we're proposing they reuse a typed name for it across engines.

## What we want from the vLLM / SGLang teams

1. **Validation of the type shape.** Does `scree.Array(values, offsets, ragged_dim=0)` capture every variant of varlen batching you do? Speak now if not — we want the primitive to fit your real workloads.

2. **Permission to draft a PR.** A small (~50 LOC) PR adding the two converter functions per engine, gated behind an optional `--scree-batch` flag so we can test without disturbing existing users.

3. **A test workload.** Ideally a CI-runnable workload (small Llama model, one short prompt) that both your engine and scree-via-your-engine can run, so we can lock in numerical equivalence in your test suite.

## Timeline (proposal)

If both teams are open to this:

- **Week 0 (now)**: this document, soliciting feedback
- **Week 2**: draft PR for vLLM, gated behind a flag
- **Week 4**: same for SGLang
- **Week 8**: numerical equivalence tests landed in both repos
- **Week 12**: optional `--scree-batch` flag exposed to users; gather feedback
- **Week 24** (post-v0.1 of scree): consider making it the default if there are no perf regressions

## Open questions

- How does vLLM's paged KV cache interact with the `scree.Array` view? Today scree treats `values` as a contiguous flat buffer; vLLM has it in HBM blocks. The bridge needs to handle non-contiguous values either by:
  1. forcing a contiguous copy at the boundary (cheap, common case), or
  2. teaching `scree.Array` about non-contiguous storage (more general, more complex)

  Open for discussion.

- Should scree gain a "view of view" type for slicing into a paged KV without copying? Useful for inference engines but adds complexity to the primitive.

## Contact

This document is `docs/vllm-integration-sketch.md` in the scree repo. Issues, comments, and counter-proposals welcome at <https://github.com/jvoltci/scree/discussions>.

— scree maintainer
