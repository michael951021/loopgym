"""JIT build + autograd wrapper for the CUDA fused log-prob kernels
(csrc/logprob_cuda.cpp + csrc/logprob_cuda_kernels.cu). Mirrors the cpp impl;
loaded only where torch.cuda.is_available() (host, not this sandbox)."""
import os
import sys

import torch

_CUDA = None
_CUDA_ERROR = None


def _load_cuda():
    """JIT-build the CUDA kernels into .tools/build (cached across processes)."""
    global _CUDA, _CUDA_ERROR
    if _CUDA is None and _CUDA_ERROR is None:
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
            _CUDA = cpp_extension.load(
                name="loopgym_logprob_cuda",
                sources=[os.path.join(root, "csrc", "logprob_cuda.cpp"),
                         os.path.join(root, "csrc", "logprob_cuda_kernels.cu")],
                build_directory=build,
                with_cuda=True,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
        except Exception as e:
            _CUDA_ERROR = e
    if _CUDA is None:
        raise RuntimeError(f"cuda kernel unavailable: {_CUDA_ERROR}")
    return _CUDA


class _CudaLogprob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, t, chunk):
        out, lse = _load_cuda().logprob_fwd(h, w, t, int(chunk))
        ctx.save_for_backward(h, w, t, lse)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        h, w, t, lse = ctx.saved_tensors
        need_h, need_w = h.requires_grad, w.requires_grad
        grad_h, grad_w = _load_cuda().logprob_bwd(h, w, t, lse, g.reshape(-1),
                                                  need_h, need_w, int(ctx.chunk))
        return grad_h if need_h else None, grad_w if need_w else None, None, None


def _cuda_impl(h, w, t, chunk):
    return _CudaLogprob.apply(h, w, t, chunk)
