// RDNA2 (gfx103x) SageAttention for the high tier's DiT attention, ported from comfy-kitchen's
// gfx103x kernel (Apache-2.0, Copyright (c) 2024 SageAttention team, Copyright (c) 2025 Comfy Org).
// Q/K stay in mossimage's existing int8 ConvRot buffers; only V needs a bf16 -> fp16 copy.
#pragma once

#include <cstdint>

#include "hip_block.h"

namespace mi::hip {

using sage_bf16 = hip_block::bf16_bits;

struct Gfx103xAttnArgs {
  const std::int8_t* __restrict__ q;         // [padded, heads * 128] int8, varlen rows
  const std::int8_t* __restrict__ k;
  const std::uint16_t* __restrict__ v;       // fp16 bits, [padded, heads * 128], native [row, head*128 + d]
  sage_bf16* __restrict__ out;               // [tokens, 3*kDim] bf16, head at head*128
  const float* __restrict__ q_scale;         // [padded, heads], one per query row
  const float* __restrict__ k_scale;         // [padded / 16, heads], one per 16 keys
  std::int64_t q_stride_n;      // row stride of q/k/v (heads * 128)
  std::int64_t q_scale_stride_n;// row stride of q_scale (heads)
  std::int64_t k_scale_stride_n;// group stride of k_scale (heads)
  std::int64_t ldo;             // output row stride (kDim)
  const std::int32_t* __restrict__ cu;       // [batch + 1] cu_seqlens on device
  int batch, max_seqlen;
  int heads;
  float sm_scale_log2e;         // (1 / sqrt(head_dim)) * log2(e)
};

// Run one sequence's attention on the current RDNA2 device. The caller must select this path only on RDNA2
// (device_is_rdna2()); non-AMD builds throw, but a non-RDNA2 AMD code object has no kernel for it.
void gfx103x_attn(const Gfx103xAttnArgs& a, hipStream_t stream);

// Convert the bf16 V rows of b.qkv + 2*kDim to fp16 bits in the layout above. cu is cu_seqlens on the device.
void gfx103x_v_to_fp16(const sage_bf16* src, std::int64_t ld, const std::int32_t* cu, std::uint16_t* dst,
                       int tokens, int batch, hipStream_t stream);

}  // namespace mi::hip
