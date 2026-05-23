# scree documentation

A cross-framework ragged tensor primitive for variable-length sequence data.

## For users

- [**Getting started**](getting-started.md) — install, first program, common patterns
- [**Concepts**](concepts.md) — the mental model behind `values + offsets + ragged_dim`
- [**API reference**](api.md) — every public function and class
- [**Bridges & migration**](bridges.md) — moving from `torch.nested`, HuggingFace, FlashAttention
- [**FAQ**](faq.md) — questions developers ask after 10 minutes with the library

## For contributors

- [**Architecture**](architecture.md) — internal layout, dispatch model, design decisions
- [**Kernels**](kernels.md) — reference and Triton kernel design, performance characteristics
- [**Benchmarks**](benchmarks.md) — methodology, reproduction, what the numbers mean
- [**Contributing**](../CONTRIBUTING.md) — how to propose changes

## Quick links

- Source on GitHub: <https://github.com/jvoltci/scree>
- PyPI: <https://pypi.org/project/scree/>
- Issues: <https://github.com/jvoltci/scree/issues>
- Discussions: <https://github.com/jvoltci/scree/discussions>
