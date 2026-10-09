// Fused token log-prob forward/backward (CUDA): same math and API as csrc/logprob_cpu.cpp;
// kernels in csrc/logprob_cuda_kernels.cu. f32 + f64 (gradcheck).
#include <torch/extension.h>

#include <algorithm>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>  // c10::cuda::getCurrentCUDAStream (no cusparse dep)

#include "logprob_cuda_kernels.h"

namespace {

void check_common(const at::Tensor& h, const at::Tensor& w, const at::Tensor& targets,
                  int64_t T_) {
  TORCH_CHECK(h.is_cuda() && w.is_cuda() && targets.is_cuda(), "logprob: CUDA only");
  TORCH_CHECK(h.is_contiguous() && w.is_contiguous() && targets.is_contiguous(),
              "logprob: inputs must be contiguous");
  TORCH_CHECK(h.dim() == 2 && w.dim() == 2, "logprob: h [T, d], w [V, d]");
  TORCH_CHECK(h.size(1) == w.size(1), "logprob: d mismatch");
  TORCH_CHECK(h.scalar_type() == w.scalar_type(), "logprob: h/w dtype mismatch");
  TORCH_CHECK(targets.dim() == 1 && targets.size(0) == h.size(0),
              "logprob: targets must be [T]");
  TORCH_CHECK(targets.scalar_type() == at::kLong, "logprob: targets must be int64");
  // A CUDA OOB write corrupts memory silently (unlike a CPU segfault).
  const int64_t V = w.size(0);
  if (T_ > 0) {
    const int64_t tmin = targets.min().item<int64_t>();
    const int64_t tmax = targets.max().item<int64_t>();
    TORCH_CHECK(tmin >= 0 && tmax < V, "logprob: targets out of range [0, V)");
  }
}

// The kernels keep per-V-lane logits in registers; chunk is capped to match.
int64_t clamp_chunk(int64_t chunk) {
  return std::min(std::max<int64_t>(chunk, 1), (int64_t)lgk::kMaxChunk);
}

}  // namespace

std::vector<at::Tensor> logprob_fwd(at::Tensor h, at::Tensor w, at::Tensor targets,
                                    int64_t chunk) {
  check_common(h, w, targets, h.size(0));
  const int T_ = (int)h.size(0);
  const int V = (int)w.size(0);
  const int d = (int)h.size(1);
  const int ch = (int)clamp_chunk(chunk);
  auto out = at::empty({h.size(0)}, h.options());
  auto lse = at::empty({h.size(0)}, h.options());
  if (T_ > 0) {
    const int64_t* tp = targets.const_data_ptr<int64_t>();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    switch (h.scalar_type()) {
      case at::kFloat:
        lgk::fwd_launch(h.const_data_ptr<float>(), w.const_data_ptr<float>(), tp,
                        out.data_ptr<float>(), lse.data_ptr<float>(), T_, V, d, ch, stream);
        break;
      case at::kDouble:
        lgk::fwd_launch(h.const_data_ptr<double>(), w.const_data_ptr<double>(), tp,
                        out.data_ptr<double>(), lse.data_ptr<double>(), T_, V, d, ch, stream);
        break;
      default:
        TORCH_CHECK(false, "logprob_fwd: unsupported dtype ", h.scalar_type());
    }
  }
  C10_CUDA_CHECK(cudaGetLastError());
  return {out, lse};
}

std::vector<at::Tensor> logprob_bwd(at::Tensor h, at::Tensor w, at::Tensor targets,
                                    at::Tensor lse, at::Tensor g, bool need_h, bool need_w,
                                    int64_t chunk) {
  check_common(h, w, targets, h.size(0));
  TORCH_CHECK(lse.is_cuda() && g.is_cuda() && lse.is_contiguous() && g.is_contiguous(),
              "logprob_bwd: lse/g must be CUDA and contiguous");
  TORCH_CHECK(lse.size(0) == h.size(0) && g.size(0) == h.size(0),
              "logprob_bwd: lse/g must be [T]");
  TORCH_CHECK(h.scalar_type() == lse.scalar_type() && h.scalar_type() == g.scalar_type(),
              "logprob_bwd: dtype mismatch");
  const int T_ = (int)h.size(0);
  const int V = (int)w.size(0);
  const int d = (int)h.size(1);
  const int ch = (int)clamp_chunk(chunk);
  // The kernels write every output element exactly once.
  auto grad_h = need_h ? at::empty_like(h) : at::Tensor();
  auto grad_w = need_w ? at::empty_like(w) : at::Tensor();
  if (T_ > 0) {
    auto g2 = g.contiguous();
    const int64_t* tp = targets.const_data_ptr<int64_t>();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    switch (h.scalar_type()) {
      case at::kFloat: {
        const float* hp = h.const_data_ptr<float>();
        const float* wp = w.const_data_ptr<float>();
        const float* lp = lse.const_data_ptr<float>();
        const float* gp = g2.const_data_ptr<float>();
        if (need_h)
          lgk::bwd_h_launch(hp, wp, tp, lp, gp, grad_h.data_ptr<float>(), T_, V, d, ch, stream);
        if (need_w)
          lgk::bwd_w_launch(hp, wp, tp, lp, gp, grad_w.data_ptr<float>(), T_, V, d, stream);
        break;
      }
      case at::kDouble: {
        const double* hp = h.const_data_ptr<double>();
        const double* wp = w.const_data_ptr<double>();
        const double* lp = lse.const_data_ptr<double>();
        const double* gp = g2.const_data_ptr<double>();
        if (need_h)
          lgk::bwd_h_launch(hp, wp, tp, lp, gp, grad_h.data_ptr<double>(), T_, V, d, ch, stream);
        if (need_w)
          lgk::bwd_w_launch(hp, wp, tp, lp, gp, grad_w.data_ptr<double>(), T_, V, d, stream);
        break;
      }
      default:
        TORCH_CHECK(false, "logprob_bwd: unsupported dtype ", h.scalar_type());
    }
  }
  C10_CUDA_CHECK(cudaGetLastError());
  return {grad_h, grad_w};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("logprob_fwd", &logprob_fwd, "chunked log-softmax log-prob forward (cuda)");
  m.def("logprob_bwd", &logprob_bwd,
        "chunked log-softmax log-prob backward (cuda); softmax from saved per-row lse");
}
