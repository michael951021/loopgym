// Fused token log-prob CUDA kernels (no torch). fwd: one block per row, online max/sum over
// V tiles so the [T, V] logits never exist. bwd: one block per row (grad_h) / per vocab row
// (grad_w); softmax recomputed from the saved per-row lse. f32 + f64 (gradcheck).
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>

#include "logprob_cuda_kernels.h"

namespace lgk {
namespace {

constexpr int kWarps = kThreads / 32;

struct SumOp {
  __device__ __forceinline__ static double combine(double a, double b) { return a + b; }
  __device__ __forceinline__ static double identity() { return 0.0; }
};
struct MaxOp {
  __device__ __forceinline__ static double combine(double a, double b) { return fmax(a, b); }
  __device__ __forceinline__ static double identity() { return -HUGE_VALL; }
};

// Block reduction over 256 threads: warp shuffles, then warp 0 over the 8 warp partials.
// Every thread returns the reduced value. `part` must be at least kThreads doubles.
template <typename Op>
__device__ double block_reduce(double x, double* part) {
  for (int o = 16; o; o >>= 1) {
    double y = __shfl_xor_sync(0xffffffffu, x, o);
    x = Op::combine(x, y);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) part[warp] = x;
  __syncthreads();
  if (warp == 0) {
    x = (lane < kWarps) ? part[lane] : Op::identity();
    for (int o = kWarps / 2; o; o >>= 1) {
      double y = __shfl_xor_sync(0xffffffffu, x, o);
      x = Op::combine(x, y);
    }
    if (lane == 0) part[0] = x;
  }
  __syncthreads();
  return part[0];
}

// Per-thread count of V lanes v = c + th + k*kThreads with v < end (chunk is clamped to
// kMaxChunk in the launchers, so nv <= kMaxVPerThread always).
__device__ __forceinline__ int lane_count(int c, int end, int th) {
  return (c + th < end) ? (end - 1 - c - th) / kThreads + 1 : 0;
}

__device__ __forceinline__ int i_min(int a, int b) { return a < b ? a : b; }

// Row dots against one V tile: s[k] = hrow . w[c + th + k*256]. j outer so a warp reads
// contiguous w addresses per j.
template <typename T>
__device__ void tile_dots(const T* __restrict__ w, const T* hrow, int c, int end, int d,
                          T* s) {
  const int th = threadIdx.x;
  const int nv = lane_count(c, end, th);
  for (int k = 0; k < nv; ++k) s[k] = T{0};
  for (int j = 0; j < d; ++j) {
    const T hj = hrow[j];
    for (int k = 0; k < nv; ++k) s[k] += hj * w[(size_t)(c + th + k * kThreads) * d + j];
  }
}

template <typename T>
__global__ void fwd_kernel(const T* __restrict__ h, const T* __restrict__ w,
                           const int64_t* __restrict__ targets, T* __restrict__ out,
                           T* __restrict__ lse, int V, int d, int chunk) {
  extern __shared__ char smem[];
  T* hrow = reinterpret_cast<T*>(smem);
  double* part = reinterpret_cast<double*>(smem + (((size_t)d * sizeof(T) + 7) & ~size_t{7}));
  const int i = blockIdx.x;
  const int th = threadIdx.x;
  for (int j = th; j < d; j += kThreads) hrow[j] = h[(size_t)i * d + j];
  __syncthreads();
  const long long tgt = targets[i];
  double m = -HUGE_VALL, S = 0.0, stgt = 0.0;
  for (int c = 0; c < V; c += chunk) {
    const int end = i_min(c + chunk, V);
    T s[kMaxVPerThread];
    tile_dots<T>(w, hrow, c, end, d, s);
    double lmax = -HUGE_VALL, lsum = 0.0;
    const int nv = lane_count(c, end, th);
    for (int k = 0; k < nv; ++k) lmax = fmax(lmax, (double)s[k]);
    const double mnew = fmax(m, block_reduce<MaxOp>(lmax, part));
    for (int k = 0; k < nv; ++k) lsum += exp((double)s[k] - mnew);
    S = S * exp(m - mnew) + block_reduce<SumOp>(lsum, part);
    m = mnew;
    const long long tv = tgt - c;
    if (tv >= 0 && tv < end - c && (tv % kThreads) == th) stgt = (double)s[tv / kThreads];
  }
  const long long ct = (tgt / chunk) * chunk;
  if ((tgt - ct) % kThreads == th) {
    const double l = m + log(S);
    out[i] = (T)(stgt - l);
    lse[i] = (T)l;
  }
}

template <typename T>
__global__ void bwd_h_kernel(const T* __restrict__ h, const T* __restrict__ w,
                             const int64_t* __restrict__ targets, const T* __restrict__ lse,
                             const T* __restrict__ g, T* __restrict__ grad_h, int V, int d,
                             int chunk) {
  extern __shared__ char smem[];
  T* hrow = reinterpret_cast<T*>(smem);
  T* acc = reinterpret_cast<T*>(smem + (((size_t)d * sizeof(T) + 7) & ~size_t{7}));
  const int i = blockIdx.x;
  const int th = threadIdx.x;
  for (int j = th; j < d; j += kThreads) {
    hrow[j] = h[(size_t)i * d + j];
    acc[j] = T{0};
  }
  __syncthreads();
  const double lse_i = (double)lse[i];
  for (int c = 0; c < V; c += chunk) {
    const int end = i_min(c + chunk, V);
    const int nv = lane_count(c, end, th);
    if (nv == 0) continue;
    T s[kMaxVPerThread];
    tile_dots<T>(w, hrow, c, end, d, s);
    T p[kMaxVPerThread];
    for (int k = 0; k < nv; ++k) p[k] = (T)exp((double)s[k] - lse_i);
    // acc[j] += sum over this thread's lanes of p * w[v, j].
    for (int j = 0; j < d; ++j) {
      T a = T{0};
      for (int k = 0; k < nv; ++k) a += p[k] * w[(size_t)(c + th + k * kThreads) * d + j];
      atomicAdd(&acc[j], a);
    }
  }
  const size_t trow = (size_t)targets[i] * d;
  for (int j = th; j < d; j += kThreads)
    grad_h[(size_t)i * d + j] = (T)((double)g[i] * ((double)w[trow + j] - (double)acc[j]));
}

template <typename T>
__global__ void bwd_w_kernel(const T* __restrict__ h, const T* __restrict__ w,
                             const int64_t* __restrict__ targets, const T* __restrict__ lse,
                             const T* __restrict__ g, T* __restrict__ grad_w, int nrows, int d) {
  extern __shared__ char smem[];
  T* wrow = reinterpret_cast<T*>(smem);
  T* acc = reinterpret_cast<T*>(smem + (((size_t)d * sizeof(T) + 7) & ~size_t{7}));
  const int v = blockIdx.x;
  const int th = threadIdx.x;
  for (int j = th; j < d; j += kThreads) {
    wrow[j] = w[(size_t)v * d + j];
    acc[j] = T{0};
  }
  __syncthreads();
  for (int t = th; t < nrows; t += kThreads) {
    double sv = 0.0;
    for (int j = 0; j < d; ++j) sv += (double)h[(size_t)t * d + j] * (double)wrow[j];
    double ds = -(double)g[t] * exp(sv - (double)lse[t]);
    if (targets[t] == v) ds += (double)g[t];
    for (int j = 0; j < d; ++j) atomicAdd(&acc[j], (T)(ds * (double)h[(size_t)t * d + j]));
  }
  for (int j = th; j < d; j += kThreads) grad_w[(size_t)v * d + j] = acc[j];
}

// smem layouts: hrow/wrow (d elements, 8-byte aligned) then either the reduction partials
// (kThreads doubles) or a second d-element accumulator.
size_t smem_fwd(int d, size_t elt) {
  return (((size_t)d * elt + 7) & ~size_t{7}) + kThreads * sizeof(double);
}
size_t smem_bwd(int d, size_t elt) {
  const size_t a = ((size_t)d * elt + 7) & ~size_t{7};
  return a + a;
}

// Raise the dynamic-smem cap monotonically (only ever raised here), then launch.
template <typename F>
void launch(const void* fn, size_t smem, F&& f) {
  static int max_smem = 0;
  if (smem > (size_t)max_smem) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    max_smem = (int)smem;
  }
  f();
}

}  // namespace

void fwd_launch(const float* h, const float* w, const int64_t* targets, float* out, float* lse,
                int T, int V, int d, int chunk, cudaStream_t stream) {
  chunk = std::min(chunk, kMaxChunk);
  const size_t smem = smem_fwd(d, sizeof(float));
  launch((const void*)fwd_kernel<float>, smem, [&] {
    fwd_kernel<float><<<T, kThreads, smem, stream>>>(h, w, targets, out, lse, V, d, chunk);
  });
}
void fwd_launch(const double* h, const double* w, const int64_t* targets, double* out,
                double* lse, int T, int V, int d, int chunk, cudaStream_t stream) {
  chunk = std::min(chunk, kMaxChunk);
  const size_t smem = smem_fwd(d, sizeof(double));
  launch((const void*)fwd_kernel<double>, smem, [&] {
    fwd_kernel<double><<<T, kThreads, smem, stream>>>(h, w, targets, out, lse, V, d, chunk);
  });
}

void bwd_h_launch(const float* h, const float* w, const int64_t* targets, const float* lse,
                  const float* g, float* grad_h, int T, int V, int d, int chunk,
                  cudaStream_t stream) {
  chunk = std::min(chunk, kMaxChunk);
  const size_t smem = smem_bwd(d, sizeof(float));
  launch((const void*)bwd_h_kernel<float>, smem, [&] {
    bwd_h_kernel<float><<<T, kThreads, smem, stream>>>(h, w, targets, lse, g, grad_h, V, d, chunk);
  });
}
void bwd_h_launch(const double* h, const double* w, const int64_t* targets, const double* lse,
                  const double* g, double* grad_h, int T, int V, int d, int chunk,
                  cudaStream_t stream) {
  chunk = std::min(chunk, kMaxChunk);
  const size_t smem = smem_bwd(d, sizeof(double));
  launch((const void*)bwd_h_kernel<double>, smem, [&] {
    bwd_h_kernel<double><<<T, kThreads, smem, stream>>>(h, w, targets, lse, g, grad_h, V, d, chunk);
  });
}

void bwd_w_launch(const float* h, const float* w, const int64_t* targets, const float* lse,
                  const float* g, float* grad_w, int T, int V, int d, cudaStream_t stream) {
  const size_t smem = smem_bwd(d, sizeof(float));
  launch((const void*)bwd_w_kernel<float>, smem, [&] {
    bwd_w_kernel<float><<<V, kThreads, smem, stream>>>(h, w, targets, lse, g, grad_w, T, d);
  });
}
void bwd_w_launch(const double* h, const double* w, const int64_t* targets, const double* lse,
                  const double* g, double* grad_w, int T, int V, int d, cudaStream_t stream) {
  const size_t smem = smem_bwd(d, sizeof(double));
  launch((const void*)bwd_w_kernel<double>, smem, [&] {
    bwd_w_kernel<double><<<V, kThreads, smem, stream>>>(h, w, targets, lse, g, grad_w, T, d);
  });
}

}  // namespace lgk
