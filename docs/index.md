---
hide:
  - navigation
  - toc
---

<div class="scree-hero" markdown>

# scree

<p class="tagline" markdown>
Variable-length tensors as a first-class type. Triton kernels at **1.6× of FlashAttention-2** on H100. Works on NumPy, PyTorch, MLX, JAX.
</p>

<div class="badges" markdown>
[![PyPI](https://img.shields.io/pypi/v/scree.svg?style=flat-square&color=1e3a5f)](https://pypi.org/project/scree/)
[![Python](https://img.shields.io/pypi/pyversions/scree.svg?style=flat-square&color=1e3a5f)](https://pypi.org/project/scree/)
[![License](https://img.shields.io/pypi/l/scree.svg?style=flat-square&color=1e3a5f)](https://github.com/jvoltci/scree/blob/master/LICENSE)
[![CI](https://github.com/jvoltci/scree/actions/workflows/ci.yml/badge.svg)](https://github.com/jvoltci/scree/actions/workflows/ci.yml)
[![Stars](https://img.shields.io/github/stars/jvoltci/scree.svg?style=social)](https://github.com/jvoltci/scree)
</div>

</div>

```python
import scree, numpy as np

# Three sequences of different lengths — no padding.
arr = scree.pack([np.random.randn(n, 8).astype(np.float32) for n in [4, 2, 7]])
# arr.values: (13, 8) ; arr.offsets: [0, 4, 6, 13]

# Run varlen attention. Each sequence attends only to itself.
from scree.kernels.reference import varlen_attention
out = varlen_attention(arr, arr, arr, causal=True)
```

[Get started →](getting-started.md){ .md-button .md-button--primary }
[GitHub →](https://github.com/jvoltci/scree){ .md-button }
[PyPI →](https://pypi.org/project/scree/){ .md-button }

## At a glance

<div class="scree-metrics" markdown>

<div class="scree-metric" markdown>
<div class="value">1.30×</div>
<div class="label">Forward vs FlashAttention-2 on H100</div>
</div>

<div class="scree-metric" markdown>
<div class="value">1.61×</div>
<div class="label">Full training step vs FA-2</div>
</div>

<div class="scree-metric" markdown>
<div class="value">85%</div>
<div class="label">Memory saved vs HF padded (inference-style)</div>
</div>

<div class="scree-metric" markdown>
<div class="value">4</div>
<div class="label">Backends: NumPy, PyTorch, MLX, JAX</div>
</div>

<div class="scree-metric" markdown>
<div class="value">68</div>
<div class="label">Tests passing across all backends</div>
</div>

<div class="scree-metric" markdown>
<div class="value">Apache-2.0</div>
<div class="label">License</div>
</div>

</div>

## Why scree

Variable-length sequences are everywhere in ML — but every team carries their own
incompatible representation. scree is the typed primitive that bridges them:

- **`torch.nested`** — PyTorch-only, beta since 2021 → `scree.bridges.to_torch_nested`
- **FlashAttention `cu_seqlens`** — convention, not a primitive → zero-copy `scree.from_cu_seqlens`
- **HuggingFace `attention_mask`** — pads then masks → bit-exact `scree.bridges.from_hf_padded`
- **vLLM / SGLang packed batches** — internal data structures → planned typed adapter
- **TF `RaggedTensor`** — TensorFlow-only → `scree.Array` is the cross-framework version

## What you can do today

| Workflow | Use case | Status |
| --- | --- | --- |
| Inference forward | Drop into your varlen attention path | ✅ 1.30× of FA-2 |
| Training step | Full backward via Triton (FA-2 style) | ✅ 1.61× of FA-2 |
| HF Transformers migration | Convert at the boundary, save 70–85% memory | ✅ Bit-exact round-trip |
| Apple Silicon training | MLX backend, native Metal kernels | ✅ |
| Cross-framework prototyping | Run the same `scree.Array` on NumPy/PyTorch/MLX/JAX | ✅ 68 tests verify agreement |

## Examples

- [01 — Quickstart](https://github.com/jvoltci/scree/blob/master/examples/01_quickstart.py): pack/unpack + varlen attention in 6 lines
- [02 — No-pad transformer](https://github.com/jvoltci/scree/blob/master/examples/02_no_pad_transformer.py): full pre-norm block, zero padding
- [03 — Training step with autograd](https://github.com/jvoltci/scree/blob/master/examples/03_train_step.py): loss drops 80× over 30 steps
- [04 — HuggingFace compat](https://github.com/jvoltci/scree/blob/master/examples/04_hf_compat.py): bit-exact migration recipe
- [05 — Multimodal interleaved](https://github.com/jvoltci/scree/blob/master/examples/05_multimodal_interleaved.py): text + image-patch sequences in one Array

## The name

A *scree* is the irregular pile of rock fragments accumulated on a mountain slope.
Variable-length sequences pack against each other the same way: irregular shapes,
fitted by their irregularity, not despite it.

---

<p style="text-align: center; opacity: 0.7; font-size: 0.9em;">
v0.0.1 on PyPI. Apache-2.0. <a href="https://github.com/jvoltci/scree">Source on GitHub</a> · <a href="faq/">FAQ</a> · <a href="https://github.com/jvoltci/scree/discussions">Discussions</a> · <a href="https://github.com/jvoltci/scree/issues">Issues</a>
</p>
