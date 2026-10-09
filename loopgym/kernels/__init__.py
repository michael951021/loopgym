"""Fused token log-probs: out[t] = log softmax(hidden[t] @ weight.T)[targets[t]].

Never materializes the [T, V] logits: the V dimension is processed in chunks with an
online max / log-sum-exp. "torch" is the reference impl every other impl must match."""
import os
import sys

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


def _torch_backward(g, h, w, t, lse, chunk):
    """Grads wrt hidden and weight; softmax recomputed per chunk, no [T, V] saved."""
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
        return _torch_backward(g, h, w, t, lse, ctx.chunk)


def _torch_impl(h, w, t, chunk):
    return _ChunkedLogprob.apply(h, w, t, chunk)


_REGISTRY = {"torch": _torch_impl}
IMPLS = ("torch",)

_CPP = None
_CPP_ERROR = None


def _load_cpp():
    """JIT-build csrc/logprob_cpu.cpp into .tools/build (cached across processes)."""
    global _CPP, _CPP_ERROR
    if _CPP is None and _CPP_ERROR is None:
        try:
            import shutil
            from torch.utils import cpp_extension
            # ninja ships in the venv bin (pyproject dep); PATH may not include it.
            if shutil.which("ninja") is None:
                os.environ["PATH"] = (os.path.dirname(sys.executable) + os.pathsep
                                      + os.environ.get("PATH", ""))
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            build = os.path.join(root, ".tools", "build")
            os.makedirs(build, exist_ok=True)
            _CPP = cpp_extension.load(
                name="loopgym_logprob_cpu",
                sources=[os.path.join(root, "csrc", "logprob_cpu.cpp")],
                build_directory=build,
                extra_cflags=["-O3", "-fopenmp"],
                extra_ldflags=["-fopenmp"],
                verbose=False,
            )
        except Exception as e:
            _CPP_ERROR = e
    if _CPP is None:
        raise RuntimeError(f"cpp kernel unavailable: {_CPP_ERROR}")
    return _CPP


class _CppLogprob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, t, chunk):
        out, lse = _load_cpp().logprob_fwd(h, w, t, int(chunk))
        ctx.save_for_backward(h, w, t, lse)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        # C++ backward lands in cppbwd; until then reuse the torch chunked math.
        h, w, t, lse = ctx.saved_tensors
        return _torch_backward(g, h, w, t, lse, ctx.chunk)


def _cpp_impl(h, w, t, chunk):
    return _CppLogprob.apply(h, w, t, chunk)


def register_impl(name, fn):
    """Register a non-torch impl (e.g. "cpp" once it builds); IMPLS is refreshed."""
    global IMPLS
    _REGISTRY[name] = fn
    IMPLS = tuple(_REGISTRY)


# Register "cpp" at import once the JIT build succeeds; no compiler -> torch-only.
try:
    _load_cpp()
    register_impl("cpp", _cpp_impl)
except Exception as _e:
    print(f"loopgym.kernels: cpp kernel not built ({_e}); torch-only", file=sys.stderr)

# Triton is usable on a GPU host or in the interpreter (sandbox, CLAUDE.md rule 2).
if os.environ.get("TRITON_INTERPRET") == "1" or torch.cuda.is_available():
    from . import triton_logprob
    register_impl("triton", triton_logprob.triton_impl)


def token_logprobs(hidden, weight, targets, impl="torch", chunk=8192):
    """Log-prob of each target token. hidden [T, d] (final hidden states), weight [V, d]
    (lm_head), targets [T] long -> [T] in hidden's dtype (float32 for float32 inputs)."""
    return _REGISTRY[impl](hidden, weight, targets, chunk)
