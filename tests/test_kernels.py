"""Regression tests for loopgym/kernels beyond the known-answer suite."""
import torch

from loopgym import kernels


def ref(h, w, t):
    return torch.log_softmax(h @ w.T, -1).gather(-1, t.unsqueeze(-1)).squeeze(-1)


def test_chunk_edge_sizes():
    g = torch.Generator().manual_seed(0)
    h, w = torch.randn(9, 16, generator=g), torch.randn(40, 16, generator=g)
    t = torch.randint(0, 40, (9,), generator=g)
    for chunk in (1, 40, 41, 9999):
        assert torch.allclose(kernels.token_logprobs(h, w, t, chunk=chunk).double(),
                              ref(h, w, t).double(), atol=1e-5)


def test_dtype_preserved_and_weight_grad():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(5, 8, dtype=torch.float64, generator=g, requires_grad=True)
    w = torch.randn(12, 8, dtype=torch.float64, generator=g, requires_grad=True)
    t = torch.randint(0, 12, (5,), generator=g)
    out = kernels.token_logprobs(h, w, t, chunk=4)
    assert out.dtype == torch.float64
    gh, gw = torch.autograd.grad(out.sum(), (h, w))
    h2, w2 = h.detach().clone().double().requires_grad_(True), w.detach().clone().double().requires_grad_(True)
    gh2, gw2 = torch.autograd.grad(ref(h2, w2, t).sum(), (h2, w2))
    assert torch.allclose(gh, gh2, atol=1e-10) and torch.allclose(gw, gw2, atol=1e-10)


def test_unknown_impl_raises():
    h, w = torch.randn(3, 4), torch.randn(5, 4)
    try:
        kernels.token_logprobs(h, w, torch.zeros(3, dtype=torch.long), impl="nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
