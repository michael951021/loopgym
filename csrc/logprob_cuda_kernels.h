// Launch wrappers for csrc/logprob_cuda_kernels.cu. Pure CUDA, no torch headers; the torch
// binding (csrc/logprob_cuda.cpp) calls these. Math mirrors csrc/logprob_cpu.cpp:
//   fwd:    out[i] = log softmax(h[i] @ w.T)[t_i], lse[i] = logsumexp (online, no [T, V])
//   bwd_h:  grad_h[i] = g[i] * (w[t_i] - sum_v p[i, v] * w[v])
//   bwd_w:  grad_w[v] = sum_t g[t] * (1[t_t == v] - p[t, v]) * h[t]
// with p recomputed per V-chunk from the saved per-row lse.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace lgk {

constexpr int kThreads = 256;
constexpr int kMaxVPerThread = 32;  // registers per thread for per-V-lane logits
constexpr int kMaxChunk = kThreads * kMaxVPerThread;  // binding clamps chunk to <= this

void fwd_launch(const float* h, const float* w, const int64_t* targets, float* out,
                float* lse, int T, int V, int d, int chunk, cudaStream_t stream);
void fwd_launch(const double* h, const double* w, const int64_t* targets, double* out,
                double* lse, int T, int V, int d, int chunk, cudaStream_t stream);

// bwd_h: one block per row (T). bwd_w: one block per vocab row (V).
void bwd_h_launch(const float* h, const float* w, const int64_t* targets, const float* lse,
                  const float* g, float* grad_h, int T, int V, int d, int chunk,
                  cudaStream_t stream);
void bwd_h_launch(const double* h, const double* w, const int64_t* targets, const double* lse,
                  const double* g, double* grad_h, int T, int V, int d, int chunk,
                  cudaStream_t stream);

void bwd_w_launch(const float* h, const float* w, const int64_t* targets, const float* lse,
                  const float* g, float* grad_w, int T, int V, int d, cudaStream_t stream);
void bwd_w_launch(const double* h, const double* w, const int64_t* targets, const double* lse,
                  const double* g, double* grad_w, int T, int V, int d, cudaStream_t stream);

}  // namespace lgk
