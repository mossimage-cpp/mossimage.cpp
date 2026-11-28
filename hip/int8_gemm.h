// Host entry point of the high-tier INT8 ConvRot GEMM (int8_gemm.hip). Kept in a header so tools can call it without
// compiling the HIP kernel source into their own translation unit.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

#include "hip_block.h"

namespace mi::hip {

// Optional fused epilogue: when `gate` is set, y[m,n] = bf16(residual[m,n] + bf16(y)*gate[n]) instead of plain y. `bias`
// and `lora_delta` are applied before the first bf16 rounding. `residual` may alias `y`. Keeping this here avoids the
// separate gate-residual kernel and its [m,n] bf16 intermediate (the o-proj and down-proj sites).
void int8_gemm(hip_block::bf16_bits* y, std::int64_t ldy, const std::int8_t* x, std::int64_t ldx, const float* x_scale,
               const std::int8_t* w, const float* w_scale, const float* bias, int m, int n, int k, hipStream_t stream,
               const hip_block::bf16_bits* lora_delta = nullptr, std::int64_t ldl = 0,
               const hip_block::bf16_bits* gate = nullptr, const hip_block::bf16_bits* residual = nullptr,
               std::int64_t ldr = 0);

}  // namespace mi::hip
