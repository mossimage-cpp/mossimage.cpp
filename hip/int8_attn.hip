// INT8 SageAttention for the DiT attention op, adapted from comfy-kitchen's gfx1103
// sage_attention kernels (Apache-2.0, Copyright (c) 2025 Comfy Org): int8 QK^T and int8 PV with
// the probabilities quantized to u8 and kept in registers, so the softmax is per-lane scalar
// work rather than a sweep through shared memory. Q is held in registers across the key loop.
//
// Only the unmasked, head_dim=128, num_kv_heads==heads case is built. The score is evaluated
// transposed (S^T = K @ Q^T), so a lane owns one query and eight keys of a 16-key MMA step;
// accumulating the online maximum and sum is a shuffle, not a block reduction.
#if defined(ROCWMMA_I8_LEAF)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "device_math.h"
#include "hip_block.h"
#include "int8_attn.h"

namespace mi::hip {
namespace {

using bf16 = hip_block::bf16_bits;
using hip_block::detail::bf;
using hip_block::detail::bits_to_float;
using hip_block::detail::f;

using v4i = int __attribute__((ext_vector_type(4)));
using v8i = int __attribute__((ext_vector_type(8)));

constexpr int kHeads = hip_block::kHeads;
constexpr int kDim = hip_block::kDim;
constexpr int kHeadDim = hip_block::kHeadDim;
constexpr int kCtaQ = 128;
constexpr int kCtaK = 64;
constexpr int kWarpQ = 16;
constexpr int kWaves = kCtaQ / kWarpQ;
constexpr int kThreads = kWaves * 32;

// Constants of the u8-Probability softmax (comfy-kitchen sage_common.h).
constexpr float kLog2e = 1.44269504088896340736f;
constexpr float kProbU8Offset = 7.9943534f;   // exp2 of this rounds to 255
constexpr float kMaskedScore = -50000.0f;
constexpr float kDroppedBias = -1.0e30f;      // unrecoverable: exp2 of it is exactly zero
constexpr float kHeadroom = 8.0f;
constexpr float kIntBiasFloat = 12582912.0f;
constexpr int kIntBias = 0x4b400000;  // bits of kIntBiasFloat; an int add becomes an fp32 fma input
static_assert(kCtaK * 128 * 255 < (1 << 22), "PV int accumulator exceeds the fp32 bias range");

__device__ __forceinline__ v4i load_b128(const void* p) { return *reinterpret_cast<const v4i*>(p); }
__device__ __forceinline__ v4i load_lds(const void* lds, int row, int kbyte, int stride) {
  return *reinterpret_cast<const v4i*>(static_cast<const char*>(lds) + row * stride + kbyte);
}

__device__ __forceinline__ int acc_row(int lane, int e) { return 2 * e + (lane >> 4); }

// gfx11's accumulator interleaves the two half-waves; swap one float between them with a
// single permute (no LDS round trip).
__device__ __forceinline__ uint32_t swap_half_wave(uint32_t v) {
  return static_cast<uint32_t>(__builtin_amdgcn_permlanex16(static_cast<int>(v), static_cast<int>(v),
                                                            0x76543210, 0xfedcba98, false, false));
}
__device__ __forceinline__ float swap_half_wave(float v) {
  return bits_to_float(swap_half_wave(hip_block::detail::float_to_bits(v)));
}

__device__ __forceinline__ uint32_t prob_to_u8(float p) {
  return static_cast<uint32_t>(fminf(255.0f, fmaxf(0.0f, rintf(p))));
}

// Rebuild a full 16-key step of u8 probabilities from the eight keys each half-wave holds.
__device__ __forceinline__ v4i pack_prob_frag(const uint32_t p[8], int lane) {
  const uint32_t lo = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
  const uint32_t hi = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);
  const uint32_t plo = swap_half_wave(lo), phi = swap_half_wave(hi);
  const bool even = lane < 16;
  const uint32_t e0 = even ? lo : plo, o0 = even ? plo : lo;
  const uint32_t e1 = even ? hi : phi, o1 = even ? phi : hi;
  v4i r;
  r[0] = static_cast<int>(__builtin_amdgcn_perm(o0, e0, 0x05010400u));
  r[1] = static_cast<int>(__builtin_amdgcn_perm(o0, e0, 0x07030602u));
  r[2] = static_cast<int>(__builtin_amdgcn_perm(o1, e1, 0x05010400u));
  r[3] = static_cast<int>(__builtin_amdgcn_perm(o1, e1, 0x07030602u));
  return r;
}

__device__ __forceinline__ uint32_t sad8(const uint32_t p[8]) {
  const uint32_t lo = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
  const uint32_t hi = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);
  return __builtin_amdgcn_sad_u8(lo, 0, __builtin_amdgcn_sad_u8(hi, 0, 0u));
}

// The block Hadamard rotation used by the q8 kernels: an in-register H4 on each group of four channels.
// Q and K get the same rotation, so dot products are unchanged and only quantization outliers move.
// The reference's full H128 was measured slower and further from the CUDA high tier's outputs, so it is not built.
__device__ __forceinline__ void convrot4(float* v) {
  const float x0 = v[0], x1 = v[1], x2 = v[2], x3 = v[3];
  const float a0 = x0 + x1, a1 = x0 - x1, a2 = x2 + x3, a3 = x2 - x3;
  v[0] = (a0 + a2) * 0.5f;
  v[1] = (a1 + a3) * 0.5f;
  v[2] = (a0 - a2) * 0.5f;
  v[3] = (a1 - a3) * 0.5f;
}

__device__ __forceinline__ void load4(const bf16* p, float* v) {
  const uint32_t lo = *reinterpret_cast<const uint32_t*>(p);
  const uint32_t hi = *reinterpret_cast<const uint32_t*>(p + 2);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    v[i] = f(static_cast<uint16_t>(lo >> (16 * i)));
    v[2 + i] = f(static_cast<uint16_t>(hi >> (16 * i)));
  }
}

// Locate the sequence of a global token and its padding shift, without any host-side metadata.
__device__ __forceinline__ void seq_of(const std::int32_t* cu, int batch, int token, int& seq, int& shift) {
  int s = 0;
  for (int i = 1; i < batch; ++i) {
    if (token >= cu[i]) s = i;
  }
  seq = s;
  shift = seq_base(cu[s], s) - cu[s];
}

__device__ __forceinline__ uint32_t quant4(const float* v, float inv_scale) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float q = fminf(127.0f, fmaxf(-127.0f, rintf(v[i] * inv_scale)));
    packed |= (static_cast<uint32_t>(static_cast<int>(q)) & 0xffu) << (8 * i);
  }
  return packed;
}

// Q: one wave per (token, head); a lane owns four adjacent channels.
__global__ __launch_bounds__(kThreads) void quant_q_kernel(const bf16* qkv, std::int64_t ld,
                                                           const std::int32_t* cu, std::int8_t* qo,
                                                           float* qs, int tokens, int heads, int batch) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int token = blockIdx.x * kWaves + wave;
  const int head = blockIdx.y;
  const bool valid = token < tokens;
  int seq = 0, shift = 0;
  if (valid) seq_of(cu, batch, token, seq, shift);
  const int col = token + shift;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (valid) load4(qkv + static_cast<std::int64_t>(token) * ld + static_cast<std::int64_t>(head) * kHeadDim +
                        lane * 4,
                   v);
  convrot4(v);
  float mx = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffull, mx, off, 32));
  const float scale = mx / 127.0f + 1e-7f;
  if (valid && lane == 0) qs[static_cast<std::size_t>(col) * heads + head] = scale;
  const uint32_t packed = quant4(v, 1.0f / scale);
  if (valid) {
    *reinterpret_cast<uint32_t*>(qo + (static_cast<std::size_t>(col) * heads + head) * kHeadDim + lane * 4) = packed;
  }
}

// K: one wave per (16-key group, head). Two passes over the group keep the rotated rows in
// registers only one at a time; the scale is uniform over the wave's 16 keys.
__global__ __launch_bounds__(kThreads) void quant_k_kernel(const bf16* qkv, std::int64_t ld,
                                                           const std::int32_t* cu, std::int8_t* ko,
                                                           float* ks, int heads) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int group = blockIdx.x * kWaves + wave;
  const int head = blockIdx.y;
  const int seq = blockIdx.z;
  const int first = cu[seq];
  const int seqlen = cu[seq + 1] - first;
  const int key0 = group * 16;
  const int col0 = seq_base(first, seq) + key0;

  float mx = 0.0f;
#pragma unroll
  for (int p = 0; p < 16; ++p) {
    if (key0 + p < seqlen) {
      float v[4];
      load4(qkv + static_cast<std::int64_t>(first + key0 + p) * ld +
                 static_cast<std::int64_t>(kDim + head * kHeadDim) + lane * 4,
            v);
      convrot4(v);
#pragma unroll
      for (int i = 0; i < 4; ++i) mx = fmaxf(mx, fabsf(v[i]));
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffull, mx, off, 32));
  const float inv_scale = 127.0f / (mx + 1e-7f);
  // A group past the sequence end has no valid keys: its amax is zero and its scale slot belongs to a later
  // sequence (or lies past the buffer). Writing it would race with that sequence's real scale or corrupt memory.
  if (key0 < seqlen && lane == 0) ks[(static_cast<std::size_t>(col0) / 16) * heads + head] = 1.0f / inv_scale;

#pragma unroll
  for (int p = 0; p < 16; ++p) {
    if (key0 + p < seqlen) {
      float v[4];
      load4(qkv + static_cast<std::int64_t>(first + key0 + p) * ld +
                 static_cast<std::int64_t>(kDim + head * kHeadDim) + lane * 4,
            v);
      convrot4(v);
      const uint32_t packed = quant4(v, inv_scale);
      *reinterpret_cast<uint32_t*>(ko + (static_cast<std::size_t>(seq_base(first, seq) + key0 + p) * heads + head) *
                                            kHeadDim +
                                        lane * 4) = packed;
    }
  }
}

// V: per-channel amax over the tokens, then quantization into the transposed [channel, key]
// buffer. Splitting the token range over blocks is what keeps this pass off the critical path.
__global__ __launch_bounds__(256) void v_amax_partial(const bf16* qkv, std::int64_t ld, float* part,
                                                      int tokens) {
  const int c = blockIdx.y * 256 + static_cast<int>(threadIdx.x);
  const int t0 = blockIdx.x * 256;
  float mx = 0.0f;
#pragma unroll
  for (int i = 0; i < 256; ++i) {
    const int token = t0 + i;
    if (token < tokens) mx = fmaxf(mx, fabsf(f(qkv[static_cast<std::int64_t>(token) * ld +
                                                 2 * kDim + c])));
  }
  part[static_cast<std::size_t>(blockIdx.x) * kHeads * kHeadDim + c] = mx;
}

__global__ __launch_bounds__(256) void v_finalize(const float* part, float* vs, int chunks) {
  const int c = blockIdx.x * 256 + static_cast<int>(threadIdx.x);
  float mx = 0.0f;
  for (int k = 0; k < chunks; ++k) mx = fmaxf(mx, part[static_cast<std::size_t>(k) * kHeads * kHeadDim + c]);
  vs[c] = fmaxf(mx / 127.0f, 1e-12f);
}

__global__ __launch_bounds__(256) void v_quant_kernel(const bf16* qkv, std::int64_t ld, const std::int32_t* cu,
                                                      const float* vs, std::int8_t* vo, int tokens, int padded,
                                                      int batch) {
  const int c = blockIdx.y * 256 + static_cast<int>(threadIdx.x);
  const int t0 = blockIdx.x * 256;
  const float inv = 1.0f / vs[c];
#pragma unroll
  for (int i = 0; i < 256; ++i) {
    const int token = t0 + i;
    if (token < tokens) {
      int seq = 0, shift = 0;
      seq_of(cu, batch, token, seq, shift);
      const float x = fminf(127.0f, fmaxf(-127.0f, rintf(f(qkv[static_cast<std::int64_t>(token) * ld +
                                                                  2 * kDim + c]) * inv)));
      vo[static_cast<std::size_t>(c) * padded + token + shift] = static_cast<std::int8_t>(x);
    }
  }
}

struct AttnArgs {
  const std::int8_t* q;
  const std::int8_t* k;
  const std::int8_t* v;
  bf16* o;
  const float* qs;
  const float* ks;
  const float* vs;
  const std::int32_t* cu;
  std::int64_t ldq;
  std::int64_t ldo;
  int heads;
  int padded;
  float scale_log2;
};

// The int8 attention below is a gfx11 WMMA kernel. RDNA2 has no WMMA, so there the DiT block uses the bf16 attention
// of block_kernels.hip (its GEMMs run on the same dot-product leaves as everything else). This trapping definition
// exists only because the launcher below is host code shared by every architecture's fat-binary object.
#if defined(__GFX10__)
__global__ __launch_bounds__(kThreads) void int8_attn_kernel(const AttnArgs) { __builtin_trap(); }
#else
__global__ __launch_bounds__(kThreads) void int8_attn_kernel(const AttnArgs a) {
  constexpr int kStrideK = kHeadDim + 16;  // 144, a multiple of 16: one ds_read_b128 per fragment
  constexpr int kStrideV = kCtaK + 16;     // 80
  __shared__ alignas(16) std::int8_t smem_k[kCtaK * kStrideK];
  __shared__ alignas(16) std::int8_t smem_v[kHeadDim * kStrideV];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int r = lane & 15;
  const int seq = static_cast<int>(blockIdx.z);
  const int head = static_cast<int>(blockIdx.y);
  const int q0 = seq_base(a.cu[seq], seq);
  const int seqlen = a.cu[seq + 1] - a.cu[seq];
  const int q_local = static_cast<int>(blockIdx.x) * kCtaQ + wave * kWarpQ + r;
  const int q_idx = q_local < seqlen ? q_local : seqlen - 1;

  // Q is loop invariant: 8 fragments of 16 bytes per lane, cached for the whole key walk.
  const std::int8_t* qrow = a.q + (static_cast<std::int64_t>(q0 + q_idx) * a.ldq + head * kHeadDim);
  v4i qf[8];
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) qf[dt] = load_b128(qrow + dt * 16);
  const float qscale = a.qs[static_cast<std::size_t>(q0 + q_idx) * a.heads + head];

  float m_run = kMaskedScore;
  float d_run = 1.0f;
  float o_bias = 0.0f;
  float oval[8][8];
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
#pragma unroll
    for (int e = 0; e < 8; ++e) oval[dt][e] = 0.0f;
  }

  const int num_iters = (seqlen + kCtaK - 1) / kCtaK;
  const int interior = a.scale_log2 >= 0.0f ? seqlen / kCtaK : 0;
  const v8i bias_acc = {kIntBias, kIntBias, kIntBias, kIntBias, kIntBias, kIntBias, kIntBias, kIntBias};

  auto run_iter = [&](auto boundary_tag, int iter) {
    constexpr bool kBoundary = decltype(boundary_tag)::value;
    const int kb0 = iter * kCtaK;
    __syncthreads();
    // K tile [key][d], zero past the sequence end.
    for (int idx = tid; idx < kCtaK * 8; idx += kThreads) {
      const int row = idx >> 3, unit = idx & 7;
      const int key = kb0 + row;
      v4i val = {0, 0, 0, 0};
      if (key < seqlen) {
        val = load_b128(a.k + (static_cast<std::int64_t>(q0 + key) * a.heads + head) * kHeadDim + unit * 16);
      }
      *reinterpret_cast<v4i*>(smem_k + row * kStrideK + unit * 16) = val;
    }
    // V tile [d][key], already transposed and zero padded by the quantizer.
    for (int idx = tid; idx < kHeadDim * 4; idx += kThreads) {
      const int d = idx >> 2, unit = idx & 3;
      const v4i val = load_b128(a.v + (static_cast<std::size_t>(head * kHeadDim + d) * a.padded + q0 + kb0 +
                                       unit * 16));
      *reinterpret_cast<v4i*>(smem_v + d * kStrideV + unit * 16) = val;
    }
    __syncthreads();

    float kscale[4];
#pragma unroll
    for (int fk = 0; fk < 4; ++fk) {
      kscale[fk] = a.ks[(static_cast<std::size_t>((q0 + kb0) / 16 + fk) * a.heads + head)];
    }

    v8i s_acc[4];
#pragma unroll
    for (int fk = 0; fk < 4; ++fk) {
      s_acc[fk] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int dt = 0; dt < 8; ++dt) {
        s_acc[fk] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, load_lds(smem_k, fk * 16 + r, dt * 16, kStrideK), true, qf[dt], s_acc[fk], false);
      }
    }

    float tile_scale_k[4];
#pragma unroll
    for (int fk = 0; fk < 4; ++fk) tile_scale_k[fk] = a.scale_log2 * qscale * kscale[fk];

    float prob[kBoundary ? 4 : 1][8];
    float tile_m = kMaskedScore;
    if constexpr (kBoundary) {
#pragma unroll
      for (int fk = 0; fk < 4; ++fk) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int key = kb0 + fk * 16 + acc_row(lane, e);
          const float score = static_cast<float>(s_acc[fk][e]) * tile_scale_k[fk];
          prob[fk][e] = key < seqlen ? score : kDroppedBias;
          tile_m = fmaxf(tile_m, prob[fk][e]);
        }
      }
    } else {
#pragma unroll
      for (int fk = 0; fk < 4; ++fk) {
        int km = s_acc[fk][0];
#pragma unroll
        for (int e = 1; e < 8; ++e) km = km > s_acc[fk][e] ? km : s_acc[fk][e];
        tile_m = fmaxf(tile_m, static_cast<float>(km) * tile_scale_k[fk]);
      }
    }
    tile_m = fmaxf(tile_m, swap_half_wave(tile_m));
    tile_m -= kProbU8Offset;

    const float m_prev = m_run;
    const bool update = tile_m > m_prev;
    const float next_m = update ? tile_m + kHeadroom : m_prev;
    if (__any(static_cast<int>(update)) != 0) {
      const float o_scale = __builtin_amdgcn_exp2f(m_prev - next_m);
      d_run *= o_scale;
      o_bias *= o_scale;
#pragma unroll
      for (int dt = 0; dt < 8; ++dt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) oval[dt][e] *= o_scale;
      }
    }
    const float tile_scale = __builtin_amdgcn_exp2f(tile_m - next_m);
    m_run = next_m;

    v4i pf[4];
    uint32_t tile_sum = 0;
#pragma unroll
    for (int fk = 0; fk < 4; ++fk) {
      uint32_t packed[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p = kBoundary ? __builtin_amdgcn_exp2f(prob[fk][e] - tile_m)
                                 : __builtin_amdgcn_exp2f(fmaf(static_cast<float>(s_acc[fk][e]),
                                                               tile_scale_k[fk], -tile_m));
        packed[e] = prob_to_u8(p);
      }
      tile_sum += sad8(packed);
      pf[fk] = pack_prob_frag(packed, lane);
    }
    d_run += static_cast<float>(tile_sum) * tile_scale;

#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      v8i partial = bias_acc;
#pragma unroll
      for (int fk = 0; fk < 4; ++fk) {
        partial = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, load_lds(smem_v, dt * 16 + r, fk * 16, kStrideV), false, pf[fk], partial, false);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        oval[dt][e] = fmaf(bits_to_float(static_cast<uint32_t>(partial[e])), tile_scale, oval[dt][e]);
      }
    }
    o_bias = fmaf(kIntBiasFloat, tile_scale, o_bias);
    if ((iter & 7) == 7) {
#pragma unroll
      for (int dt = 0; dt < 8; ++dt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) oval[dt][e] -= o_bias;
      }
      o_bias = 0.0f;
    }
  };

  int iter = 0;
  for (; iter < interior; ++iter) run_iter(std::false_type{}, iter);
  for (; iter < num_iters; ++iter) run_iter(std::true_type{}, iter);

  d_run += swap_half_wave(d_run);
  if (q_local < seqlen) {
    const float inv_d = 1.0f / d_run;
    // Q/K/V live in the padded, per-sequence columns, but the output is the compact [tokens, kDim] attn buffer.
    // Writing at q0 would run past it once batch > 1 (the padded gaps) and corrupt the following buffers.
    bf16* orow = a.o + (static_cast<std::int64_t>(a.cu[seq]) + q_local) * a.ldo + head * kHeadDim;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int d = dt * 16 + acc_row(lane, e);
        orow[d] = bf((oval[dt][e] - o_bias) * a.vs[head * kHeadDim + d] * inv_d);
      }
    }
  }
}
#endif  // !__GFX10__

void fail(const char* what, hipError_t e) {
  throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

}  // namespace

void int8_attn_quantize(const bf16* qkv, std::int64_t ld, const Int8AttnBuffers& b, int tokens,
                        hipStream_t stream, bool quant_v) {
  const dim3 qgrid(static_cast<unsigned>((tokens + kWaves - 1) / kWaves),
                   static_cast<unsigned>(kHeads), 1);
  quant_q_kernel<<<qgrid, kThreads, 0, stream>>>(qkv, ld, b.cu, b.q, b.q_scale, tokens, kHeads, b.batch);

  const int groups = b.padded / 16;
  const dim3 kgrid(static_cast<unsigned>((groups + kWaves - 1) / kWaves),
                   static_cast<unsigned>(kHeads), static_cast<unsigned>(b.batch));
  quant_k_kernel<<<kgrid, kThreads, 0, stream>>>(qkv, ld, b.cu, b.k, b.k_scale, kHeads);

  if (quant_v) {
    const int chunks = (tokens + 255) / 256;
    const dim3 vgrid(static_cast<unsigned>(chunks), static_cast<unsigned>(kHeads * kHeadDim / 256), 1);
    // No memset of the V padding: keys past the sequence end get zero probability in the attention
    // kernel, so their uninitialized V bytes never contribute to PV.
    v_amax_partial<<<vgrid, 256, 0, stream>>>(qkv, ld, b.part, tokens);
    v_finalize<<<dim3(static_cast<unsigned>(kHeads * kHeadDim / 256)), 256, 0, stream>>>(b.part, b.v_scale, chunks);
    v_quant_kernel<<<vgrid, 256, 0, stream>>>(qkv, ld, b.cu, b.v_scale, b.v, tokens, b.padded, b.batch);
  }

  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) fail("int8_attn_quantize", e);
}

void int8_attn_run(const Int8AttnBuffers& b, bf16* out, std::int64_t ldo, int max_seqlen,
                   float scale, hipStream_t stream) {
  const AttnArgs a{b.q, b.k, b.v, out, b.q_scale, b.k_scale, b.v_scale, b.cu,
                   static_cast<std::int64_t>(kHeads * kHeadDim), ldo, kHeads, b.padded, scale * kLog2e};
  const dim3 grid(static_cast<unsigned>((max_seqlen + kCtaQ - 1) / kCtaQ),
                  static_cast<unsigned>(kHeads), static_cast<unsigned>(b.batch));
  int8_attn_kernel<<<grid, kThreads, 0, stream>>>(a);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) fail("int8_attn_run", e);
}

}  // namespace mi::hip

#else  // !ROCWMMA_I8_LEAF (the NVIDIA logic-check build has no gfx11 int8 WMMA)

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "int8_attn.h"

namespace mi::hip {

void int8_attn_quantize(const i8attn_bf16*, std::int64_t, const Int8AttnBuffers&, int, hipStream_t, bool) {
  throw std::runtime_error("int8 attention needs a gfx11 int8 WMMA build");
}
void int8_attn_run(const Int8AttnBuffers&, i8attn_bf16*, std::int64_t, int, float, hipStream_t) {
  throw std::runtime_error("int8 attention needs a gfx11 int8 WMMA build");
}

}  // namespace mi::hip

#endif
