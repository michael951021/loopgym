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

// Backward: grad_h[t] = g[t] * (W[t_t] - sum_v p[t, v] W[v]);
// grad_w[v] = sum_t g[t] (1[v == t_t] - p[t, v]) h[t]; p = softmax recomputed per
// V-chunk from the saved per-row lse, so no [T, V] tensor exists.
template <typename T>
std::vector<at::Tensor> logprob_bwd_impl(const at::Tensor& h, const at::Tensor& w,
                                         const at::Tensor& targets, const at::Tensor& lse,
                                         const at::Tensor& g, bool need_h, bool need_w,
                                         int64_t chunk) {
  const int64_t T_ = h.size(0);
  const int64_t V = w.size(0);
  chunk = std::max<int64_t>(chunk, 1);
  auto grad_h = need_h ? at::zeros_like(h) : at::Tensor();
  auto grad_w = need_w ? at::zeros_like(w) : at::Tensor();
  const int64_t* tp = targets.const_data_ptr<int64_t>();
  auto g2 = g.contiguous();
  const T* gp = g2.data_ptr<T>();
  for (int64_t c = 0; c < V; c += chunk) {
    const int64_t ck = std::min<int64_t>(chunk, V - c);
    auto w_c = w.slice(0, c, c + ck);
    auto s = at::matmul(h, w_c.t());
    auto dS = -at::exp(s - lse.unsqueeze(1)) * g2.unsqueeze(1);  // -g * p per chunk
    T* dsp = dS.data_ptr<T>();
    for (int64_t i = 0; i < T_; ++i) {
      const int64_t vt = tp[i];
      if (vt >= c && vt < c + ck) dsp[i * ck + (vt - c)] += gp[i];
    }
    if (need_h) grad_h.addmm_(dS, w_c);
    if (need_w) grad_w.slice(0, c, c + ck) = dS.t().mm(h);
  }
  return {grad_h, grad_w};
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

std::vector<at::Tensor> logprob_bwd(at::Tensor h, at::Tensor w, at::Tensor targets,
                                    at::Tensor lse, at::Tensor g, bool need_h, bool need_w,
                                    int64_t chunk) {
  TORCH_CHECK(h.is_cpu() && w.is_cpu() && targets.is_cpu() && lse.is_cpu() && g.is_cpu(),
              "logprob_bwd: CPU only");
  TORCH_CHECK(h.dim() == 2 && w.dim() == 2 && h.size(1) == w.size(1),
              "logprob_bwd: h [T, d], w [V, d]");
  TORCH_CHECK(lse.size(0) == h.size(0) && g.size(0) == h.size(0) && targets.size(0) == h.size(0),
              "logprob_bwd: lse/g/targets must be [T]");
  TORCH_CHECK(h.scalar_type() == w.scalar_type() && h.scalar_type() == lse.scalar_type() &&
                  h.scalar_type() == g.scalar_type(),
              "logprob_bwd: dtype mismatch");
  switch (h.scalar_type()) {
    case at::kFloat:
      return logprob_bwd_impl<float>(h, w, targets, lse, g, need_h, need_w, chunk);
    case at::kDouble:
      return logprob_bwd_impl<double>(h, w, targets, lse, g, need_h, need_w, chunk);
    default:
      TORCH_CHECK(false, "logprob_bwd: unsupported dtype ", h.scalar_type());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("logprob_fwd", &logprob_fwd, "chunked log-softmax log-prob forward (cpu)");
  m.def("logprob_bwd", &logprob_bwd,
        "chunked log-softmax log-prob backward (cpu); softmax from saved per-row lse");
}
