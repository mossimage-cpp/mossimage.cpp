// The high tier's INT8 ConvRot DiT block: the four linears (qkv, o, up_gate, down) as int8 x int8 -> int32 -> bf16, the
// activations rotated+quantized per row before each linear, the norms/modulation/rope/attention/SiTU-GLU in bf16. The op
// order and the bf16 rounding points are the engine's (models/moss_dit/dit_block_quant, all-INT8, no LoRA) —
// hip_block::run_block_op with the four GEMMs swapped for int8 GEMMs plus a bf16 epilogue.
#pragma once

#include <cstdint>

#include "hip_block.h"
#include "int8_attn.h"

namespace mi::hip {

using bf16 = hip_block::bf16_bits;

// One rotated+quantized linear: W [n, k] int8 (row-major, the ConvRot q) and one per-row float scale [n].
struct Int8Linear {
  const std::int8_t* q;
  const float* scale;
  const float* bias;  // [n], null for no bias
};

// One INT8 ConvRot block's weights. The bf16 norms feed the shared bf16 sub-ops; the four linears are Int8Linear.
struct Int8BlockWeights {
  const bf16* norm1;  // [kDim]
  const bf16* norm2;  // [kDim]
  const bf16* q_norm; // [128]
  const bf16* k_norm; // [128]
  Int8Linear qkv;     // [3 kDim, kDim]
  Int8Linear o;       // [kDim, kDim]
  Int8Linear up_gate; // [2 kFfnDim, kDim]
  Int8Linear down;    // [kDim, kFfnDim]
};

// The INT8 block's scratch, sized for one run of `tokens` tokens. i8_y/i8_ffn hold a row's rotated+quantized int8, scale
// its per-row scale, ug the up_gate GEMM's bf16 output. The o/down GEMMs write their fused gate+residual result in place.
struct Int8Buffers {
  std::int8_t* i8_y;      // [tokens, kDim] int8
  std::int8_t* i8_ffn;    // [tokens, kFfnDim] int8
  float* scale;           // [tokens]
  bf16* ug;               // [tokens, 2 kFfnDim] bf16
  Int8AttnBuffers attn;   // quantized q/k/v and scales for the int8 attention op
};

// One run of `tokens` tokens; b's x, y, qkv, attn, ffn, rope, mod, cu_seqlens, batch, max_seqlen are the shared bf16
// buffers (hip_block::BlockBuffers). Same op order as hip_block::run_block_op.
void run_int8_block_op(int op, const Int8BlockWeights& w, const hip_block::BlockBuffers& b, const Int8Buffers& ib,
                       hipStream_t stream, const bf16* lora_delta = nullptr, std::int64_t ldl = 0,
                       bool fuse_situ = false);

}  // namespace mi::hip
