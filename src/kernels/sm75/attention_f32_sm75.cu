// Exact attention on Turing cards without tensor cores. The sm_75 flash kernel executes its m16n8k8 fp16 MMA on
// ordinary CUDA cores (~0.3 TFLOPS). Here the two matmuls are cuBLASLt fp32 GEMMs (0.7-1.5 TFLOPS on TU117) and the
// softmax is a SIMT kernel; scores are chunked one head at a time so the score matrix stays in the caller's workspace.
// Q/K/V are bf16 on input; converting them to fp32 is exact, and the output is rounded once to bf16.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/cuda_libs.h"
#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::sm75 {
namespace {

constexpr std::int64_t kAlign = 256;
std::size_t pad(std::size_t b) { return (b + kAlign - 1) / kAlign * kAlign; }
std::uint32_t alignment(const void* p) {
  auto a = reinterpret_cast<std::uintptr_t>(p);
  std::uint32_t v = 256;
  while (v > 1 && a % v != 0) v /= 2;
  return v;
}

void check_lt(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("attention_f32: cuBLASLt ") + what + ": " + cublasLtGetStatusString(s));
  }
}

// Row-major D[M,N] = X[M,K]·W[N,K]ᵀ. Plans are cached per shape: the layouts must outlive every captured graph that
// uses them, so they are never destroyed between calls.
class F32Gemm {
 public:
  F32Gemm() {
    check_lt(cublasLtCreate(&h_), "create");
    check_lt(cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta), "transa");
    check_lt(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb), "transb");
  }
  ~F32Gemm() {
    for (auto& kv : plans_) {
      if (kv.second.d != nullptr) cublasLtMatrixLayoutDestroy(kv.second.d);
      if (kv.second.b != nullptr) cublasLtMatrixLayoutDestroy(kv.second.b);
      if (kv.second.a != nullptr) cublasLtMatrixLayoutDestroy(kv.second.a);
    }
    if (desc_ != nullptr) cublasLtMatmulDescDestroy(desc_);
    if (h_ != nullptr) cublasLtDestroy(h_);
  }
  F32Gemm(const F32Gemm&) = delete;
  F32Gemm& operator=(const F32Gemm&) = delete;

  void run(const void* w, const void* x, void* d, int m, int n, int k, int ldx, int ldw, int ldo, void* ws,
           cudaStream_t stream) {
    const std::array<int, 6> key{m, n, k, ldx, ldw, ldo};
    // Plans are never erased, so references into the map stay valid after another thread inserts a plan. Lock only
    // the cache access; the matmul itself may run concurrently on different streams.
    const Plan* plan = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = plans_.find(key);
      if (it == plans_.end()) {
        Plan p;
        check_lt(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_32F, k, n, ldw), "layout A");
        check_lt(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_32F, k, m, ldx), "layout B");
        check_lt(cublasLtMatrixLayoutCreate(&p.d, CUDA_R_32F, n, m, ldo), "layout D");
        cublasLtMatmulPreference_t pref = nullptr;
        check_lt(cublasLtMatmulPreferenceCreate(&pref), "preference");
        struct Guard {
          cublasLtMatmulPreference_t p;
          ~Guard() { cublasLtMatmulPreferenceDestroy(p); }
        } guard{pref};
        const std::size_t ws_bytes = std::size_t{32} << 20;
        check_lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                                                      sizeof ws_bytes), "workspace pref");
        const std::uint32_t schemes = CUBLASLT_REDUCTION_SCHEME_NONE;
        check_lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &schemes,
                                                      sizeof schemes), "reduction mask");
        const std::uint32_t aw = alignment(w), ax = alignment(x), ad = alignment(d);
        check_lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &aw, sizeof aw),
                 "align A");
        check_lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &ax, sizeof ax),
                 "align B");
        check_lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &ad, sizeof ad),
                 "align C");
        check_lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &ad, sizeof ad),
                 "align D");
        cublasLtMatmulHeuristicResult_t r{};
        int found = 0;
        check_lt(cublasLtMatmulAlgoGetHeuristic(h_, desc_, p.a, p.b, p.d, p.d, pref, 1, &r, &found), "heuristic");
        if (found == 0) throw std::runtime_error("attention_f32: no cuBLASLt algorithm");
        p.algo = r.algo;
        p.ws = r.workspaceSize;
        it = plans_.emplace(key, p).first;
      }
      plan = &it->second;
    }
    const float alpha = 1.f, beta = 0.f;
    check_lt(cublasLtMatmul(h_, desc_, &alpha, w, plan->a, x, plan->b, &beta, d, plan->d, d, plan->d, &plan->algo, ws,
                            plan->ws, stream), "matmul");
  }

 private:
  struct Plan {
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algo{};
    std::size_t ws = 0;
  };
  cublasLtHandle_t h_ = nullptr;
  cublasLtMatmulDesc_t desc_ = nullptr;
  std::mutex mutex_;
  std::map<std::array<int, 6>, Plan> plans_;
};

// Convert one head's q/k/v rows from their padded bf16 views to contiguous fp32 [len, dim] arrays.
__global__ void convert_qkv_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v, std::int64_t q_row, std::int64_t q_head,
                                   std::int64_t k_row, std::int64_t k_head, std::int64_t v_row, std::int64_t v_head,
                                   int head, int kv_head, int dim, int len_q, int len_k, int q_start, int k_start,
                                   float* __restrict__ qf, float* __restrict__ kf, float* __restrict__ vf) {
  const std::int64_t total = static_cast<std::int64_t>(len_q + len_k) * dim;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    if (i < static_cast<std::int64_t>(len_q) * dim) {
      const int t = static_cast<int>(i / dim), d = static_cast<int>(i % dim);
      qf[i] = __bfloat162float(q[static_cast<std::int64_t>(q_start + t) * q_row + head * q_head + d]);
    } else {
      const int j = static_cast<int>((i - static_cast<std::int64_t>(len_q) * dim) / dim);
      const int d = static_cast<int>(i % dim);
      kf[static_cast<std::int64_t>(j) * dim + d] =
          __bfloat162float(k[static_cast<std::int64_t>(k_start + j) * k_row + kv_head * k_head + d]);
      vf[static_cast<std::int64_t>(j) * dim + d] =
          __bfloat162float(v[static_cast<std::int64_t>(k_start + j) * v_row + kv_head * v_head + d]);
    }
  }
}

// Row-wise softmax of the raw QK^T scores, in place; one thread per row (in-place P must not be read by peers).
// causal_shift = len_k - len_q (bottom-right FA2 alignment). Writes LSE in the engine's [head, token] layout.
__global__ void softmax_kernel(float* __restrict__ s, int len_k, int causal_shift, bool causal, float scale,
                               float* __restrict__ lse, std::int64_t lse_head_off, int q_start) {
  __shared__ float shared[16];  // [0..7]: warp maxima; [8..15]: warp sums
  const int row = blockIdx.x, tid = threadIdx.x;
  float* in = s + static_cast<std::int64_t>(row) * len_k;
  int allowed = len_k;
  if (causal) allowed = std::min(len_k, std::max(0, row + causal_shift + 1));
  float local_max = -INFINITY;
  for (int j = tid; j < allowed; j += blockDim.x) local_max = fmaxf(local_max, in[j] * scale);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, off));
  if ((tid & 31) == 0) shared[tid >> 5] = local_max;
  __syncthreads();
  if (tid == 0) {
    float mx = -INFINITY;
    for (int w = 0; w < blockDim.x / 32; ++w) mx = fmaxf(mx, shared[w]);
    shared[0] = mx;
  }
  __syncthreads();
  const float mx = shared[0];
  if (mx == -INFINITY) {
    for (int j = tid; j < len_k; j += blockDim.x) in[j] = 0.f;
    if (tid == 0) lse[lse_head_off + q_start + row] = INFINITY;
    return;
  }
  float local_sum = 0.f;
  for (int j = tid; j < len_k; j += blockDim.x) {
    const float p = j < allowed ? __expf(in[j] * scale - mx) : 0.f;
    in[j] = p;
    local_sum += p;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_down_sync(0xffffffffu, local_sum, off);
  if ((tid & 31) == 0) shared[8 + (tid >> 5)] = local_sum;
  __syncthreads();
  if (tid == 0) {
    float sum = 0.f;
    for (int w = 0; w < blockDim.x / 32; ++w) sum += shared[8 + w];
    shared[1] = sum;
    lse[lse_head_off + q_start + row] = mx + logf(sum);
  }
  __syncthreads();
  const float inv = 1.f / shared[1];
  for (int j = tid; j < len_k; j += blockDim.x) in[j] *= inv;
}

// V is [len_k, dim]; cuBLASLt's linear computes out = x·wᵀ, so PV needs w = Vᵀ [dim, len_k].
__global__ void transpose_v_kernel(const float* __restrict__ vf, float* __restrict__ vt, int len_k, int dim) {
  const std::int64_t total = static_cast<std::int64_t>(len_k) * dim;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int j = static_cast<int>(i / dim), d = static_cast<int>(i % dim);
    vt[static_cast<std::int64_t>(d) * len_k + j] = vf[i];
  }
}

__global__ void store_out_kernel(const float* __restrict__ of, __nv_bfloat16* __restrict__ out, std::int64_t o_row,
                                 std::int64_t o_head, int head, int dim, int len_q, int q_start) {
  const std::int64_t total = static_cast<std::int64_t>(len_q) * dim;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(i / dim), d = static_cast<int>(i % dim);
    out[static_cast<std::int64_t>(q_start + t) * o_row + head * o_head + d] = __float2bfloat16_rn(of[i]);
  }
}

template <class F>
void grid_stride(F&& fn, std::int64_t total, cudaStream_t stream) {
  if (total <= 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>((total + threads - 1) / threads, 65535));
  fn(blocks, threads, stream);
}

struct Scratch {
  float* lse;
  void* lt_ws;
  float *q, *k, *v, *vt, *o, *s;
};

// `max_seqlen` is an upper bound on one sequence's key length; varlen runs with reference slots have len_k > len_q,
// so k/v, Vᵀ and the score matrix must be sized from it rather than from the total query count.
Scratch slice(void* workspace, std::int64_t total_q, int heads, int dim, std::int64_t max_seqlen) {
  auto* const base = static_cast<std::byte*>(workspace);
  Scratch r;
  std::size_t off = 0;
  r.lse = reinterpret_cast<float*>(base + off);
  off = pad(off + static_cast<std::size_t>(total_q) * heads * sizeof(float));
  r.lt_ws = base + off;
  off = pad(off + (std::size_t{32} << 20));
  const std::int64_t buf_tokens = std::max<std::int64_t>(total_q, max_seqlen);
  const std::size_t per = pad(static_cast<std::size_t>(buf_tokens) * dim * sizeof(float));
  r.q = reinterpret_cast<float*>(base + off); off += per;
  r.k = reinterpret_cast<float*>(base + off); off += per;
  r.v = reinterpret_cast<float*>(base + off); off += per;
  r.o = reinterpret_cast<float*>(base + off); off += per;
  r.vt = reinterpret_cast<float*>(base + off);
  off = pad(off + pad(static_cast<std::size_t>(dim) * max_seqlen * sizeof(float)));
  r.s = reinterpret_cast<float*>(base + off);
  return r;
}

}  // namespace

std::size_t flash_fwd_workspace_bytes_sm75(std::int64_t total_q, int heads, int dim, std::int64_t max_seqlen) {
  if (total_q <= 0) return 0;
  const std::size_t per =
      pad(static_cast<std::size_t>(std::max<std::int64_t>(total_q, max_seqlen)) * dim * sizeof(float));
  return pad(static_cast<std::size_t>(total_q) * heads * sizeof(float)) + pad(std::size_t{32} << 20) + 4 * per +
         pad(static_cast<std::size_t>(dim) * max_seqlen * sizeof(float)) +
         pad(static_cast<std::size_t>(total_q) * max_seqlen * sizeof(float));
}

void flash_fwd_f32(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& cu_seqlens_q,
                   const Tensor& cu_seqlens_k, int max_seqlen, bool causal, float scale, void* workspace,
                   std::size_t workspace_bytes, cudaStream_t stream) {
  if (max_seqlen <= 0) throw std::runtime_error("flash_fwd_f32: max_seqlen must be positive");
  const std::int64_t total_q = q.dim(0);
  if (total_q == 0) return;
  // cuBLASLt's fp32 kernels read the matrices past their logical bounds; the matrix views live on the caller's
  // arena stack, so let the stream drain before the GEMM reuses that memory. Without this, WDDM/Turing drivers
  // surface an illegal access in the GEMM (seen as "attention_f32: cuBLASLt matmul ...").
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  const int heads = q.dim(1), dim = q.dim(2);
  if (dim != 128 && dim != 256) throw std::runtime_error("flash_fwd_f32: unsupported head dim");
  if (workspace == nullptr ||
      workspace_bytes < flash_fwd_workspace_bytes_sm75(total_q, heads, dim, max_seqlen)) {
    throw std::runtime_error("flash_fwd_f32: workspace too small for the fp32 attention path");
  }
  mi::backend::load_cublaslt();
  static F32Gemm gemm;
  const Scratch ws = slice(workspace, total_q, heads, dim, max_seqlen);
  const int batch = cu_seqlens_q.dim(0) - 1;
  std::vector<int> hu(batch + 1), hk(batch + 1);
  MI_CUDA_CHECK(cudaMemcpy(hu.data(), cu_seqlens_q.data, (batch + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  MI_CUDA_CHECK(cudaMemcpy(hk.data(), cu_seqlens_k.data, (batch + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  const auto* qp = q.ptr<const __nv_bfloat16>();
  const auto* kp = k.ptr<const __nv_bfloat16>();
  const auto* vp = v.ptr<const __nv_bfloat16>();
  auto* op = out.ptr<__nv_bfloat16>();
  const int kv_heads = k.dim(1);
  const int group = heads / kv_heads;

  for (int b = 0; b < batch; ++b) {
    const int q0 = hu[b], len_q = hu[b + 1] - q0, k0 = hk[b], len_k = hk[b + 1] - k0;
    if (len_q == 0 || len_k == 0) continue;
    const int shift = len_k - len_q;
    for (int h = 0; h < heads; ++h) {
      const int kh = h / group;
      grid_stride([&](int blocks, int threads, cudaStream_t s) {
        convert_qkv_kernel<<<blocks, threads, 0, s>>>(qp, kp, vp, q.stride[0], q.stride[1], k.stride[0], k.stride[1],
                                                       v.stride[0], v.stride[1], h, kh, dim, len_q, len_k, q0, k0,
                                                       ws.q, ws.k, ws.v);
      }, static_cast<std::int64_t>(len_q + len_k) * dim, stream);

      const Tensor sq = Tensor::view(ws.q, DType::kF32, Device::kCUDA, {len_q, dim});
      const Tensor sk = Tensor::view(ws.k, DType::kF32, Device::kCUDA, {len_k, dim});
      const Tensor ss = Tensor::view(ws.s, DType::kF32, Device::kCUDA, {len_q, len_k});
      gemm.run(sk.data, sq.data, ss.data, len_q, len_k, dim, sq.stride[0], sk.stride[0], ss.stride[0], ws.lt_ws,
               stream);

      softmax_kernel<<<len_q, 256, 0, stream>>>(ws.s, len_k, shift, causal, scale, ws.lse,
                                                static_cast<std::int64_t>(h) * total_q, q0);

      grid_stride([&](int blocks, int threads, cudaStream_t s) {
        transpose_v_kernel<<<blocks, threads, 0, s>>>(ws.v, ws.vt, len_k, dim);
      }, static_cast<std::int64_t>(len_k) * dim, stream);
      const Tensor sp = Tensor::view(ws.s, DType::kF32, Device::kCUDA, {len_q, len_k});
      const Tensor svt = Tensor::view(ws.vt, DType::kF32, Device::kCUDA, {dim, len_k});
      const Tensor so = Tensor::view(ws.o, DType::kF32, Device::kCUDA, {len_q, dim});
      gemm.run(svt.data, sp.data, so.data, len_q, dim, len_k, sp.stride[0], svt.stride[0], so.stride[0], ws.lt_ws,
               stream);

      grid_stride([&](int blocks, int threads, cudaStream_t s) {
        store_out_kernel<<<blocks, threads, 0, s>>>(ws.o, op, out.stride[0], out.stride[1], h, dim, len_q, q0);
      }, static_cast<std::int64_t>(len_q) * dim, stream);
    }
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::sm75
