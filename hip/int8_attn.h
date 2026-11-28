// INT8 SageAttention for the high tier's DiT attention op. Port of comfy-kitchen's gfx1103
// sage_attention kernels (int8 QK + int8 PV, softmax probabilities held as u8), simplified to
// this backend's needs: head_dim 128, non-causal, num_kv_heads == heads, varlen sequences.
//
// The three quantized operands are laid out for the WMMA fragment format and the LDS tiles:
//   q, k: [padded, heads * 128] int8, row stride heads * 128, contiguous channels;
//   v:    [heads * 128, padded] int8, row stride padded, TRANSPOSED (row = channel, col = key);
//   q_scale: [padded, heads] float, one scale per query row;
//   k_scale: [padded / 16, heads] float, one scale per 16 adjacent keys;
//   v_scale: [heads * 128] float, one scale per channel (applied to the output).
// `padded` rounds each sequence's start/end out to a 64-key tile and inserts a 64-key gap
// between sequences, so every key tile is in bounds. The V padding is left uninitialized: out-of-range
// keys get zero probability, so those bytes never contribute to PV.
#pragma once

#include <cstddef>
#include <cstdint>

#include "hip_block.h"

namespace mi::hip {

using i8attn_bf16 = hip_block::bf16_bits;

struct Int8AttnBuffers {
  std::int8_t* q = nullptr;       // [padded, kHeads * kHeadDim]
  std::int8_t* k = nullptr;       // [padded, kHeads * kHeadDim]
  std::int8_t* v = nullptr;       // [kHeads * kHeadDim, padded]
  float* q_scale = nullptr;       // [padded, kHeads]
  float* k_scale = nullptr;       // [padded / 16, kHeads]
  float* v_scale = nullptr;       // [kHeads * kHeadDim]
  float* part = nullptr;          // [ceil(tokens / 256), kHeads * kHeadDim] partial V amax
  std::uint16_t* v_fp16 = nullptr; // RDNA2: fp16 V, [padded, kHeads * kHeadDim], same row stride as q/k
  const std::int32_t* cu = nullptr;  // [batch + 1] cu_seqlens of this run (already on the device)
  int batch = 0;
  int padded = 0;
};

// Number of padded key columns needed for `tokens` tokens in `batch` sequences.
inline std::size_t int8_attn_padded(int tokens, int batch) {
  const std::size_t t = static_cast<std::size_t>(tokens);
  return ((t + 63u) / 64u) * 64u + 64u * static_cast<std::size_t>(batch);
}

// Extra scratch for the RDNA2 attention's fp16 V copy.
inline std::size_t int8_attn_fp16_v_bytes(int tokens, int batch) {
  return int8_attn_padded(tokens, batch) * static_cast<std::size_t>(hip_block::kHeads) * hip_block::kHeadDim * 2u;
}

inline std::size_t int8_attn_bytes(int tokens, int batch) {
  const std::size_t padded = int8_attn_padded(tokens, batch);
  const std::size_t heads = static_cast<std::size_t>(hip_block::kHeads);
  const std::size_t dim = static_cast<std::size_t>(hip_block::kHeadDim);
  return 2u * padded * heads * dim +           // q, k
         padded * heads * dim +                // v (transposed)
         padded * heads * sizeof(float) +      // q_scale
         (padded / 16u) * heads * sizeof(float) +  // k_scale
         heads * dim * sizeof(float) +         // v_scale
         ((static_cast<std::size_t>(tokens) + 255u) / 256u) * heads * dim * sizeof(float) +  // part
          (static_cast<std::size_t>(batch) + 1u) * sizeof(std::int32_t);  // cu_seqlens
}

// Sequence b starts at column seq_base(cu[b], b): cu rounded out to a 64-key tile plus a 64-key
// gap per preceding sequence. Everything downstream derives its offsets from cu on the device,
// so no host-side metadata upload is needed between layers.
constexpr int seq_base(int cu_b, int b) { return ((cu_b + 63) / 64) * 64 + 64 * b; }

// Quantize q/k/v out of the [tokens, 3 * kDim] qkv tensor into the buffers above.
// b.cu and b.padded must already be set; per-sequence base, length and position offsets are derived from cu on device.
void int8_attn_quantize(const i8attn_bf16* qkv, std::int64_t ld, const Int8AttnBuffers& b, int tokens,
                        hipStream_t stream, bool quant_v = true);

// Run attention, writing bf16 [tokens, ldo] with head columns 128 apart.
void int8_attn_run(const Int8AttnBuffers& b, i8attn_bf16* out, std::int64_t ldo, int max_seqlen,
                   float scale, hipStream_t stream);

}  // namespace mi::hip
