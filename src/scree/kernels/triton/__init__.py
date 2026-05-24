"""Triton GPU kernels for scree.

These kernels require a CUDA-capable device and Triton installed
(``pip install triton``). Import is gated so this package can be loaded
on platforms without CUDA (CPU-only, Apple Silicon, etc.) without
raising at import time — the actual kernels raise informative errors
only when invoked.
"""

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    from ._autograd import varlen_attention_triton_autograd
    from .varlen_attention import varlen_attention_triton
    from .varlen_layernorm import varlen_layernorm_triton
    from .varlen_rmsnorm import varlen_rmsnorm_triton

    __all__ = [
        "varlen_attention_triton",
        "varlen_attention_triton_autograd",
        "varlen_layernorm_triton",
        "varlen_rmsnorm_triton",
        "TRITON_AVAILABLE",
    ]
else:
    __all__ = ["TRITON_AVAILABLE"]
