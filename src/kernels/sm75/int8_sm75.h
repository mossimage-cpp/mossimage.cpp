// sm_75 INT8 ConvRot GEMMs: native tiled __dp4a GEMMs, with the same int32 accumulation and dequantisation order as
// the sm_80 CUTLASS kernels. Gated-residual and SiTU-GLU run the GEMM into a temporary and apply the bit-exact SIMT
// epilogue; the runtime LoRA side branch can add its delta in the GEMM epilogue.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "core/tensor.h"
#include "kernels/quant/quant.h"

namespace mi::kernels::sm75 {

// Same contracts as kernels/quant/quant.h's int8_gemm / int8_gemm_gated_residual / int8_gemm_situ_glu (operand checks,
// output layout and numerics); the host-side dispatcher in those translation units routes compute capability 7.x here.
void int8_gemm(const quant::Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream);
// Same as int8_gemm, but adds a bf16 delta in the GEMM epilogue (the runtime LoRA side branch).
void int8_gemm_add_delta(const quant::Int8Operands& op, const __nv_bfloat16* delta, std::int64_t ldd,
                         __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream);
void int8_gemm_gated_residual(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream);
void int8_gemm_situ_glu(const quant::Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream);

}  // namespace mi::kernels::sm75
