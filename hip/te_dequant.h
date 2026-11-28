// Device-side dequantization of the text encoder's and the DiT's extra int8 ConvRot weights.
//
// Each row of g=256 int8 values is scaled and multiplied by H_256 = kron^4(H4) as four radix-4 butterfly stages,
// reproducing dequant_fast.h's host transform operation for operation and rounded once to bf16 at the end. The int8
// weights stay on the device (no second, bf16 host image and its copy); only the scales are uploaded once at
// construction. The bf16 result differs from quant::dequantize_rows in <0.05 % of the bits, by at most one ulp.
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>

namespace mi::hip::te_dequant {

// q: [n, k] int8, scale: [n] fp32, out: [n, k] bf16 bits; k a multiple of 256.
void dequant_i8(const std::int8_t* q, const float* scale, std::uint16_t* out, int n, int k, hipStream_t stream);

}  // namespace mi::hip::te_dequant
