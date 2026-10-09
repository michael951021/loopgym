// Fused token log-prob forward (CPU): out[t] = log softmax(h[t] @ w.T)[t[t]],
// lse[t] = logsumexp(h[t] @ w.T). V is walked in chunks with an online max/sum so the
// [T, V] logits are never materialized; rows are split across OpenMP threads. float32
// and float64 (gradcheck); half types are for the CUDA/Triton impls.
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <omp.h>

namespace {

template <typename T>
std::vector<at::Tensor> logprob_fwd_impl(const at::Tensor& h, const at::Tensor& w,
                                         const at::Tensor& targets, int64_t chunk) {
  const int64_t T_ = h.size(0);
  const int64_t V = w.size(0);
  chunk = std::max<int64_t>(chunk, 1);
  auto out = at::empty({T_}, h.options());
  auto lse = at::empty({T_}, h.options());
  const int64_t* tp = targets.const_data_ptr<int64_t>();
  T* op = out.data_ptr<T>();
  T* lp = lse.data_ptr<T>();
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < T_; ++i) {
    double m = -std::numeric_limits<double>::infinity();
    double S = 0.0;
    double tgt = 0.0;
    for (int64_t c = 0; c < V; c += chunk) {
      const int64_t ck = std::min<int64_t>(chunk, V - c);
      // Row dot products against this V-chunk: [ck].
      auto s = at::matmul(h[i], w.slice(0, c, c + ck).t());
      const double mn = std::max(m, s.amax().item<double>());
      // Double accumulation keeps f32 inputs within the 1e-4 known-test tolerance.
      S = S * std::exp(m - mn) + at::sum(at::exp(s - mn), at::kDouble).item<double>();
      m = mn;
      const int64_t vt = tp[i];
      if (vt >= c && vt < c + ck)
        tgt = s[vt - c].item<double>();
    }
    const double l = m + std::log(S);
    op[i] = static_cast<T>(tgt - l);
    lp[i] = static_cast<T>(l);
  }
  return {out, lse};
}

}  // namespace

std::vector<at::Tensor> logprob_fwd(at::Tensor h, at::Tensor w, at::Tensor targets,
                                    int64_t chunk) {
  TORCH_CHECK(h.is_cpu() && w.is_cpu() && targets.is_cpu(), "logprob_fwd: CPU only");
  TORCH_CHECK(h.dim() == 2 && w.dim() == 2, "logprob_fwd: h [T, d], w [V, d]");
  TORCH_CHECK(h.size(1) == w.size(1), "logprob_fwd: d mismatch");
  TORCH_CHECK(h.scalar_type() == w.scalar_type(), "logprob_fwd: h/w dtype mismatch");
  TORCH_CHECK(targets.dim() == 1 && targets.size(0) == h.size(0),
              "logprob_fwd: targets must be [T]");
  TORCH_CHECK(targets.scalar_type() == at::kLong, "logprob_fwd: targets must be int64");
  switch (h.scalar_type()) {
    case at::kFloat:
      return logprob_fwd_impl<float>(h, w, targets, chunk);
    case at::kDouble:
      return logprob_fwd_impl<double>(h, w, targets, chunk);
    default:
      TORCH_CHECK(false, "logprob_fwd: unsupported dtype ", h.scalar_type());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("logprob_fwd", &logprob_fwd,
                                                 "chunked log-softmax log-prob forward (cpu)"); }
