"""Tests verifying that PyTorch autograd flows through scree.Array.

The scree.Array dataclass is just a typed view — gradients live on the
underlying ``values`` tensor and propagate normally. These tests lock in
that guarantee so refactors don't accidentally break backprop.
"""

import pytest

torch = pytest.importorskip("torch")

import scree
from scree.kernels.reference import varlen_attention, varlen_layernorm, varlen_rmsnorm


def test_grad_flows_through_pack_unpack():
    seqs = [torch.randn(n, 4, requires_grad=True) for n in [3, 5, 2]]
    arr = scree.pack(seqs)
    out = scree.unpack(arr)
    loss = sum(t.sum() for t in out)
    loss.backward()
    for s in seqs:
        assert s.grad is not None
        assert s.grad.shape == s.shape
        assert torch.isfinite(s.grad).all()


def test_grad_flows_through_varlen_attention():
    seqs_q = [torch.randn(n, 2, 4, requires_grad=True) for n in [3, 5]]
    seqs_k = [torch.randn(n, 2, 4, requires_grad=True) for n in [3, 5]]
    seqs_v = [torch.randn(n, 2, 4, requires_grad=True) for n in [3, 5]]
    q = scree.pack(seqs_q)
    k = scree.pack(seqs_k)
    v = scree.pack(seqs_v)
    out = varlen_attention(q, k, v, causal=True)
    out.values.sum().backward()
    for s in seqs_q + seqs_k + seqs_v:
        assert s.grad is not None
        assert s.grad.shape == s.shape
        assert torch.isfinite(s.grad).all()


def test_grad_flows_through_varlen_rmsnorm_with_weight():
    seqs = [torch.randn(n, 8) for n in [3, 5]]
    arr = scree.pack(seqs)
    weight = torch.ones(8, requires_grad=True)
    out = varlen_rmsnorm(arr, weight=weight, eps=1e-6)
    out.values.sum().backward()
    assert weight.grad is not None
    assert weight.grad.shape == weight.shape
    assert torch.isfinite(weight.grad).all()


def test_grad_flows_through_varlen_layernorm_with_weight_bias():
    seqs = [torch.randn(n, 8) for n in [3, 5]]
    arr = scree.pack(seqs)
    weight = torch.ones(8, requires_grad=True)
    bias = torch.zeros(8, requires_grad=True)
    out = varlen_layernorm(arr, weight=weight, bias=bias, eps=1e-5)
    out.values.sum().backward()
    assert weight.grad is not None
    assert bias.grad is not None
    assert torch.isfinite(weight.grad).all()
    assert torch.isfinite(bias.grad).all()


def test_grad_finite_for_short_sequences():
    """Edge case: length-1 sequences (no attention to mask)."""
    seqs_q = [torch.randn(1, 1, 4, requires_grad=True) for _ in range(3)]
    seqs_k = [torch.randn(1, 1, 4, requires_grad=True) for _ in range(3)]
    seqs_v = [torch.randn(1, 1, 4, requires_grad=True) for _ in range(3)]
    q = scree.pack(seqs_q)
    k = scree.pack(seqs_k)
    v = scree.pack(seqs_v)
    out = varlen_attention(q, k, v, causal=True)
    out.values.sum().backward()
    for s in seqs_q + seqs_k + seqs_v:
        assert torch.isfinite(s.grad).all()
