#include "kernels/lora/lora_kernels.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/cuda_check.h"

namespace mi::kernels::lora {
namespace {

template <class T>
__global__ void scale_columns_kernel(const float* __restrict__ t32, std::int64_t ld32, const float* __restrict__ coef,
                                     T* __restrict__ t, std::int64_t ldt, std::int64_t m, std::int64_t r) {
  const std::int64_t total = m * r;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t row = i / r, col = i % r;
    const float v = __fmul_rn(t32[row * ld32 + col], coef[col]);
    if constexpr (std::is_same_v<T, float>) {
      t[row * ldt + col] = v;
    } else {
      t[row * ldt + col] = __float2bfloat16_rn(v);
    }
  }
}

void run_scale(const char* op, const float* t32, std::int64_t ld32, const float* coef, void* t, std::int64_t ldt,
               std::int64_t m, std::int64_t r, cudaStream_t stream, bool f32) {
  if (m < 0 || r <= 0 || ld32 < r || ldt < r || t32 == nullptr || coef == nullptr || t == nullptr) {
    throw std::invalid_argument(std::string(op) + ": bad arguments (m " + std::to_string(m) + ", r " +
                                std::to_string(r) + ", ld32 " + std::to_string(ld32) + ", ldt " +
                                std::to_string(ldt) + ")");
  }
  if (m == 0) return;
  constexpr int kThreads = 256;
  const std::int64_t blocks = std::min<std::int64_t>((m * r + kThreads - 1) / kThreads, 4096);
  if (f32) {
    scale_columns_kernel<float><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        t32, ld32, coef, static_cast<float*>(t), ldt, m, r);
  } else {
    scale_columns_kernel<__nv_bfloat16><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        t32, ld32, coef, static_cast<__nv_bfloat16*>(t), ldt, m, r);
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

void scale_columns_bf16(const float* t32, std::int64_t ld32, const float* coef, __nv_bfloat16* t, std::int64_t ldt,
                        std::int64_t m, std::int64_t r, cudaStream_t stream) {
  run_scale("scale_columns_bf16", t32, ld32, coef, t, ldt, m, r, stream, false);
}

void scale_columns_f32(const float* t32, std::int64_t ld32, const float* coef, float* t, std::int64_t ldt,
                       std::int64_t m, std::int64_t r, cudaStream_t stream) {
  run_scale("scale_columns_f32", t32, ld32, coef, t, ldt, m, r, stream, true);
}

}  // namespace mi::kernels::lora
