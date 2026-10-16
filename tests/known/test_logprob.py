"""Fused token log-probs (loopgym/kernels): log softmax(hidden @ weight.T)[target] without the [T, V] logits.
Every impl in loopgym.kernels.IMPLS must pass; "torch" is the plain reference."""
import pytest
import torch

from loopgym import kernels


def reference(h, w, t):
    return torch.log_softmax((h @ w.T).double(), -1).gather(-1, t.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("impl", kernels.IMPLS)
@pytest.mark.parametrize("T,V,d,chunk", [(7, 50, 16, 8), (33, 1000, 64, 256), (5, 3001, 32, 1024)])
def test_matches_reference(impl, T, V, d, chunk):
    g = torch.Generator().manual_seed(T * V)
    h, w = torch.randn(T, d, generator=g), torch.randn(V, d, generator=g) * 0.5
    t = torch.randint(0, V, (T,), generator=g)
    out = kernels.token_logprobs(h, w, t, impl=impl, chunk=chunk)
    assert out.shape == (T,) and out.dtype == torch.float32
    assert torch.allclose(out.double(), reference(h, w, t), atol=1e-4)


@pytest.mark.parametrize("impl", kernels.IMPLS)
def test_large_logits_are_stable(impl):
    h = torch.tensor([[100.0, 0.0], [0.0, -100.0]])
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]) * 10
    t = torch.tensor([0, 2])
    out = kernels.token_logprobs(h, w, t, impl=impl, chunk=2)
    assert torch.isfinite(out).all() and torch.allclose(out.double(), reference(h, w, t), atol=1e-3)


@pytest.mark.parametrize("impl", kernels.IMPLS)
def test_gradient_wrt_hidden(impl):
    g = torch.Generator().manual_seed(1)
    h = torch.randn(6, 8, dtype=torch.float64, generator=g, requires_grad=True)
    w = torch.randn(40, 8, dtype=torch.float64, generator=g)
    t = torch.randint(0, 40, (6,), generator=g)
    assert torch.autograd.gradcheck(lambda x: kernels.token_logprobs(x, w, t, impl=impl, chunk=16).double(), (h,), atol=1e-4)


@pytest.mark.model
@pytest.mark.parametrize("impl", kernels.IMPLS)
def test_real_model_matches_hf_loss(impl):
    """On the real Qwen2.5-Coder-0.5B (V = 151936): mean of -logprob of each next token == HF's own loss."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from known_paths import MODEL
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    ids = tok("def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n", return_tensors="pt").input_ids
    with torch.no_grad():
        out = m(ids, labels=ids, output_hidden_states=True)
        h = out.hidden_states[-1][0]                     # final hidden states (after the last norm)
        lp = kernels.token_logprobs(h[:-1], m.get_output_embeddings().weight, ids[0, 1:], impl=impl)
    assert abs(float(-lp.mean()) - float(out.loss)) < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("impl", [i for i in kernels.IMPLS if i != "torch"])
def test_runs_on_a_non_current_gpu(impl):
    """The GRPO trainer lives on cuda:1 while cuda:0 is current; kernels must follow the tensors."""
    g = torch.Generator().manual_seed(3)
    h = torch.randn(5, 64, generator=g).to("cuda:1").requires_grad_()
    w = torch.randn(300, 64, generator=g).to("cuda:1").requires_grad_()
    t = torch.randint(0, 300, (5,), generator=g).to("cuda:1")
    assert torch.cuda.current_device() == 0
    lp = kernels.token_logprobs(h, w, t, impl=impl, chunk=64)
    ref = kernels.token_logprobs(h, w, t, impl="torch", chunk=64)
    assert torch.allclose(lp, ref, atol=1e-4)
    lp.sum().backward()
    assert h.grad is not None and h.grad.device == h.device
