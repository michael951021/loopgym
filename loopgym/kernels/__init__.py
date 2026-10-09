"""Fused token log-probs: out[t] = log softmax(hidden[t] @ weight.T)[targets[t]].

Never materializes the [T, V] logits: the V dimension is processed in chunks with an
online max / log-sum-exp. "torch" is the reference impl every other impl must match."""
import torch

__all__ = ["IMPLS", "token_logprobs", "register_impl"]


def _chunked_forward(h, w, t, chunk):
    """Returns (out, lse) in h's dtype. lse[t] = logsumexp(h[t] @ w.T)."""
    T, _ = h.shape
    V = w.shape[0]
    dev, dt = h.device, h.dtype
    m = torch.full((T,), float("-inf"), dtype=dt, device=dev)
    S = torch.zeros(T, dtype=dt, device=dev)
    tl = torch.empty(T, dtype=dt, device=dev)
    for c in range(0, V, chunk):
        w_c = w[c:c + chunk]
        s = h @ w_c.T
        m_new = torch.maximum(m, s.amax(dim=-1))
        S = S * torch.exp(m - m_new) + torch.exp(s - m_new.unsqueeze(-1)).sum(dim=-1)
        m = m_new
        in_c = (t >= c) & (t < c + w_c.shape[0])
        tl[in_c] = s[in_c, t[in_c] - c]
    lse = m + torch.log(S)
    return tl - lse, lse


class _ChunkedLogprob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, t, chunk):
        out, lse = _chunked_forward(h, w, t, chunk)
        ctx.save_for_backward(h, w, t, lse)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        h, w, t, lse = ctx.saved_tensors
        chunk = ctx.chunk
        T, _ = h.shape
        V = w.shape[0]
        g = g.reshape(-1)
        grad_h = torch.zeros_like(h) if h.requires_grad else None
        grad_w = torch.zeros_like(w) if w.requires_grad else None
        rows = torch.arange(T, device=h.device)
        for c in range(0, V, chunk):
            w_c = w[c:c + chunk]
            s = h @ w_c.T
            # dS[t, v] = g[t] * (1[v == t_t] - softmax[s][t, v]); recompute softmax per chunk.
            dS = -g.unsqueeze(-1) * torch.exp(s - lse.unsqueeze(-1))
            in_c = (t >= c) & (t < c + w_c.shape[0])
            if in_c.any():
                dS[rows[in_c], t[in_c] - c] += g[in_c]
            if grad_h is not None:
                grad_h += dS @ w_c
            if grad_w is not None:
                grad_w[c:c + w_c.shape[0]] = dS.T @ h
        return grad_h, grad_w, None, None


def _torch_impl(h, w, t, chunk):
    return _ChunkedLogprob.apply(h, w, t, chunk)


_REGISTRY = {"torch": _torch_impl}
IMPLS = ("torch",)


def register_impl(name, fn):
    """Register a non-torch impl (e.g. "cpp" once it builds); IMPLS is refreshed."""
    global IMPLS
    _REGISTRY[name] = fn
    IMPLS = tuple(_REGISTRY)


def token_logprobs(hidden, weight, targets, impl="torch", chunk=8192):
    """Log-prob of each target token. hidden [T, d] (final hidden states), weight [V, d]
    (lm_head), targets [T] long -> [T] in hidden's dtype (float32 for float32 inputs)."""
    return _REGISTRY[impl](hidden, weight, targets, chunk)
