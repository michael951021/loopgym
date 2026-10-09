"""Triton impl of token log-probs: V-tiled online max/log-sum-exp, dtype-preserving.

Forward: one program per row, tiles over V (online max + sum-exp; no [T, V] logits).
Backward: bwd_h (one program per row, accumulates E_p[w]) and bwd_w (one program per
V tile, loops rows) recompute the per-tile softmax; no [T, V] saved, no atomics.
Checked under TRITON_INTERPRET=1 in the sandbox (rule 2); GPU run via scripts/gpu_run.sh."""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(H, W, T_idx, Out, LSE, V, d, C: tl.constexpr, DC: tl.constexpr):
    row = tl.program_id(0)
    dc = tl.arange(0, DC)
    h = tl.load(H + row * d + dc, mask=dc < d, other=0.0)
    tv = tl.load(T_idx + row)
    m = float("-inf")
    S = 0.0
    targ = float("-inf")
    for c in range(0, V, C):
        offs = c + tl.arange(0, C)
        mask = offs < V
        w_c = tl.load(W + offs[:, None] * d + dc[None, :],
                      mask=mask[:, None] & (dc[None, :] < d), other=0.0)
        s = tl.where(mask, tl.sum(w_c * h[None, :], axis=1), float("-inf"))
        m_new = tl.maximum(m, tl.max(s))
        S = S * tl.exp(m - m_new) + tl.sum(tl.exp(s - m_new))
        m = m_new
        targ = tl.maximum(targ, tl.max(tl.where(mask & (offs == tv), s, float("-inf"))))
    tl.store(Out + row, targ - (m + tl.log(S)))
    tl.store(LSE + row, m + tl.log(S))


@triton.jit
def _bwd_h(H, W, G, LSE, T_idx, GradH, V, d, C: tl.constexpr, DC: tl.constexpr):
    row = tl.program_id(0)
    dc = tl.arange(0, DC)
    h = tl.load(H + row * d + dc, mask=dc < d, other=0.0)
    g = tl.load(G + row)
    lse = tl.load(LSE + row)
    tv = tl.load(T_idx + row)
    acc = tl.zeros([DC], dtype=h.dtype)
    for c in range(0, V, C):
        offs = c + tl.arange(0, C)
        mask = offs < V
        w_c = tl.load(W + offs[:, None] * d + dc[None, :],
                      mask=mask[:, None] & (dc[None, :] < d), other=0.0)
        s = tl.where(mask, tl.sum(w_c * h[None, :], axis=1), float("-inf"))
        p = tl.exp(s - lse)
        acc += tl.sum(p[:, None] * w_c, axis=0)
    w_t = tl.load(W + tv * d + dc, mask=dc < d, other=0.0)
    tl.store(GradH + row * d + dc, g * (w_t - acc), mask=dc < d)


@triton.jit
def _bwd_w(H, W, G, LSE, T_idx, GradW, T, V, d, C: tl.constexpr, DC: tl.constexpr):
    c = tl.program_id(0) * C
    dc = tl.arange(0, DC)
    offs = c + tl.arange(0, C)
    mask = offs < V
    dm = mask[:, None] & (dc[None, :] < d)
    w_c = tl.load(W + offs[:, None] * d + dc[None, :], mask=dm, other=0.0)
    acc = tl.zeros([C, DC], dtype=H.dtype.element_ty)
    for row in range(0, T):
        h = tl.load(H + row * d + dc, mask=dc < d, other=0.0)
        g = tl.load(G + row)
        lse = tl.load(LSE + row)
        tv = tl.load(T_idx + row)
        s = tl.where(mask, tl.sum(w_c * h[None, :], axis=1), float("-inf"))
        p = tl.exp(s - lse)
        dS = g * (tl.where(offs == tv, 1.0, 0.0) - p)
        acc += dS[:, None] * h[None, :]
    tl.store(GradW + offs[:, None] * d + dc[None, :], acc, mask=dm)


class _TritonLogprob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, t, chunk):
        T, _ = h.shape
        V = w.shape[0]
        out = torch.empty(T, dtype=h.dtype, device=h.device)
        lse = torch.empty(T, dtype=h.dtype, device=h.device)
        if T > 0:
            _fwd[(T,)](h, w, t, out, lse, V, h.shape[1], C=chunk,
                       DC=triton.next_power_of_2(h.shape[1]))
        ctx.save_for_backward(h, w, t, lse)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        h, w, t, lse = ctx.saved_tensors
        chunk = ctx.chunk
        T, d = h.shape
        V = w.shape[0]
        g = g.reshape(-1)
        dc = triton.next_power_of_2(d)
        grad_h = torch.empty_like(h) if h.requires_grad else None
        grad_w = torch.empty_like(w) if w.requires_grad else None
        if T > 0:
            if grad_h is not None:
                _bwd_h[(T,)](h, w, g, lse, t, grad_h, V, d, C=chunk, DC=dc)
            if grad_w is not None:
                _bwd_w[(triton.cdiv(V, chunk),)](h, w, g, lse, t, grad_w, T, V, d, C=chunk, DC=dc)
        return grad_h, grad_w, None, None


def triton_impl(h, w, t, chunk):
    """Same contract as the torch impl; V-tile size rounded down to a power of two."""
    c = 2 ** (max(1, chunk).bit_length() - 1)
    if os.environ.get("TRITON_INTERPRET") == "1":
        # interpreter caps one block at 1048576 elements (triton/_utils.py)
        c = min(c, 1024)
    return _TritonLogprob.apply(h, w, t, c)


