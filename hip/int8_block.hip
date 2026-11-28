// The high tier's INT8 ConvRot DiT block (int8_block.h): the four linears (qkv, o, up_gate, down) as int8 x int8 ->
// int32 -> bf16 with per-row activation and weight scales, the activations rotated+quantized per row before each linear,
// and the norms, modulation, RoPE, attention and SiTU-GLU in bf16. The op order and the bf16 rounding points follow the
// engine's all-INT8 block (models/moss_dit/dit_block_quant, no LoRA), which is hip_block::run_block_op with the four
// GEMMs swapped for int8 GEMMs plus a bf16 epilogue.
#include "int8_block.h"

#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device_math.h"
#include "dit_kernels.h"
#include "runtime.h"
#include "int8_attn.h"
#include "int8_gemm.h"
#include "sage_attn_gfx103x.h"
#include "rotate_quantize.hip"

namespace mi::hip {
namespace {

using namespace hip_block;
using detail::add;
using detail::bf;
using detail::f;
using detail::mul;

constexpr std::int64_t kDim = 3072;
constexpr std::int64_t kQkv = 3 * 3072;
constexpr std::int64_t kFfn = 8192;
constexpr int kHeads = 24;
constexpr std::int64_t kHead = 128;

}  // namespace

void run_int8_block_op(int op, const Int8BlockWeights& w, const BlockBuffers& b, const Int8Buffers& ib, hipStream_t stream,
                       const bf16* lora_delta, std::int64_t ldl, bool fuse_situ) {
  const std::int64_t tokens = b.tokens;
  const auto mod = [&](int i) { return b.mod + static_cast<std::int64_t>(i) * kDim; };
  switch (op) {
    case 0:  // norm1 + modulate -> y (bf16)
      rms_norm_modulate(b.x, kDim, w.norm1, mod(1), mod(0), b.y, kDim, tokens, kDim, stream);
      break;
    case 1:  // qkv = int8_gemm(rotate_quantize(y))
      rotate_quantize(b.y, kDim, tokens, kDim, ib.i8_y, kDim, ib.scale, stream);
      int8_gemm(b.qkv, kQkv, ib.i8_y, kDim, ib.scale, w.qkv.q, w.qkv.scale, nullptr,
                static_cast<int>(tokens), static_cast<int>(kQkv), static_cast<int>(kDim), stream, lora_delta, ldl);
      break;
    case 2:  // qk norm + rope (in place on qkv)
      qk_norm_rope(b.qkv, kQkv, tokens, kHeads, w.q_norm, w.k_norm, b.rope, kHead, stream);
      break;
    case 3:  // attention: int8 QK + int8 PV on gfx11; bf16 attention on RDNA2 (no WMMA for the int8 SageAttention)
      if (device_is_rdna2()) {
        // The comfy-kitchen gfx103x int8 attention: Q/K come from the same quantized buffers as the gfx11 path,
        // V is copied to fp16, and cu_seqlens is read on device so each sequence keeps its own padded columns.
        // The gfx103x kernel reads Q/K in int8 and V in fp16, so skip the int8 V quantizer's three kernels.
        int8_attn_quantize(b.qkv, kQkv, ib.attn, static_cast<int>(tokens), stream, false);
        gfx103x_v_to_fp16(b.qkv, kQkv, b.cu_seqlens, ib.attn.v_fp16, static_cast<int>(tokens), b.batch, stream);
        gfx103x_attn({ib.attn.q, ib.attn.k, ib.attn.v_fp16, b.attn, ib.attn.q_scale, ib.attn.k_scale,
                      kHeads * kHead, kHeads, kHeads, kDim, b.cu_seqlens, b.batch, b.max_seqlen, kHeads,
                      static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHead)) * 1.4426950408889634)},
                     stream);
      } else {
        int8_attn_quantize(b.qkv, kQkv, ib.attn, static_cast<int>(tokens), stream);
        int8_attn_run(ib.attn, b.attn, kDim, b.max_seqlen,
                      static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHead))), stream);
      }
      break;
    case 4:  // x += int8_gemm(rotate_quantize(attn)) * gate_msa
      rotate_quantize(b.attn, kDim, tokens, kDim, ib.i8_y, kDim, ib.scale, stream);
      int8_gemm(b.x, kDim, ib.i8_y, kDim, ib.scale, w.o.q, w.o.scale, nullptr,
                static_cast<int>(tokens), static_cast<int>(kDim), static_cast<int>(kDim), stream, lora_delta, ldl,
                mod(2), b.x, kDim);  // gate+residual fused: no gemm_out round trip
      break;
    case 5:  // norm2 + modulate -> y (bf16)
      rms_norm_modulate(b.x, kDim, w.norm2, mod(4), mod(3), b.y, kDim, tokens, kDim, stream);
      break;
    case 6:  // ffn = situ_glu(int8_gemm(rotate_quantize(y)))
      rotate_quantize(b.y, kDim, tokens, kDim, ib.i8_y, kDim, ib.scale, stream);
      int8_gemm(ib.ug, kFfn * 2, ib.i8_y, kDim, ib.scale, w.up_gate.q, w.up_gate.scale, nullptr,
                static_cast<int>(tokens), static_cast<int>(kFfn * 2), static_cast<int>(kDim), stream, lora_delta, ldl);
      if (!fuse_situ) situ_glu(ib.ug, kFfn * 2, b.ffn, kFfn, tokens, kFfn, stream);
      break;
    default:  // x += int8_gemm(rotate_quantize(ffn)) * gate_mlp
      if (fuse_situ)
        rotate_quantize_situ(ib.ug, kFfn * 2, tokens, kFfn, ib.i8_ffn, kFfn, ib.scale, stream);
      else
        rotate_quantize(b.ffn, kFfn, tokens, kFfn, ib.i8_ffn, kFfn, ib.scale, stream);
      int8_gemm(b.x, kDim, ib.i8_ffn, kFfn, ib.scale, w.down.q, w.down.scale, nullptr,
                static_cast<int>(tokens), static_cast<int>(kDim), static_cast<int>(kFfn), stream, lora_delta, ldl,
                mod(5), b.x, kDim);  // gate+residual fused: no gemm_out round trip
      break;
  }
}

}  // namespace mi::hip
