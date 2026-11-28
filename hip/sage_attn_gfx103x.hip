// RDNA2 SageAttention for the DiT attention, ported from comfy-kitchen's gfx103x kernel.
// Only the attention itself is ported: mossimage's Q/K quantizers (which include its ConvRot4) and its padded varlen
// buffers are kept, so the kernel takes strides for the scale buffers and applies sm_scale once on the Q scale.
#if defined(ROCWMMA_I8_LEAF)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "int8_attn.h"
#include "sage_attn_gfx103x.h"

namespace mi::hip {
namespace {

constexpr int kHeadDim = 128;
constexpr int kBlockM = 128;   // queries per block, one thread per query
constexpr int kBlockN = 16;    // keys per tile
constexpr int kTM = 8;         // output columns per thread group
constexpr int kQuads = kHeadDim / 4;
constexpr int kTD = kHeadDim / kTM;
constexpr int kMinBlkK = 16;
constexpr float kMaskedScore = -3.0e38f;

__device__ __forceinline__ int sdot4(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, true); }
__device__ __forceinline__ float fdot2(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2(*reinterpret_cast<const _Float16 __attribute__((ext_vector_type(2)))*>(&a),
                                *reinterpret_cast<const _Float16 __attribute__((ext_vector_type(2)))*>(&b), c, true);
}
__device__ __forceinline__ sage_bf16 float_to_bf16(float v) {
  const std::uint32_t u = __float_as_uint(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// [padded, heads * 128] fp16 V from the bf16 qkv rows [tokens, 3*kDim]. The padded column gap is left uninitialized:
// keys past the true sequence length get probability zero and never contribute to P*V.
__global__ void v_to_fp16_kernel(const sage_bf16* qkv, std::int64_t ld, const std::int32_t* __restrict__ cu,
                                 std::uint16_t* dst, int tokens, int batch) {
  const int idx = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
  if (idx >= tokens * kHeadDim * hip_block::kHeads) return;
  const int d = idx % kHeadDim, h = (idx / kHeadDim) % hip_block::kHeads, token = idx / (kHeadDim * hip_block::kHeads);
  int seq = 0;
  for (int s = 1; s < batch; ++s) if (token >= cu[s]) seq = s;
  const int col = seq_base(cu[seq], seq) + (token - cu[seq]);
  const sage_bf16 b = qkv[static_cast<std::int64_t>(token) * ld + 2 * hip_block::kDim + h * kHeadDim + d];
  dst[static_cast<std::size_t>(col) * hip_block::kHeads * kHeadDim + h * kHeadDim + d] =
      __half_as_ushort(__float2half(hip_block::detail::f(b)));
}

#if defined(__GFX10__)

__global__ __attribute__((amdgpu_waves_per_eu(2))) __launch_bounds__(kBlockM, 1) void attn_kernel(
    const Gfx103xAttnArgs a) {
  constexpr int KS_TOTAL = kBlockN * kHeadDim, VS_TOTAL = kHeadDim * kBlockN;
  __shared__ __attribute__((aligned(32))) std::int8_t k_buf[2][KS_TOTAL];
  __shared__ __attribute__((aligned(32))) __half v_buf[2][VS_TOTAL];
  __shared__ __attribute__((aligned(32))) __half p_buf[kBlockN * kBlockM];
  __shared__ __attribute__((aligned(32))) float alpha_buf[kBlockM], l_buf[kBlockM];

  const int tid = static_cast<int>(threadIdx.x);
  const int seq = static_cast<int>(blockIdx.z);
  const int first = a.cu[seq], len = a.cu[seq + 1] - first;
  const int col0 = seq_base(first, seq);  // padded key/query column of this sequence
  const int m = static_cast<int>(blockIdx.x) * kBlockM + tid;
  const int h = static_cast<int>(blockIdx.y);
  const bool valid = m < len;

  float qsv = 0.0f;
  int q_reg[kQuads];
  if (valid) {
    qsv = a.q_scale[static_cast<std::int64_t>(col0 + m) * a.q_scale_stride_n + h] * a.sm_scale_log2e;
    const std::int8_t* qrow = a.q + static_cast<std::int64_t>(col0 + m) * a.q_stride_n + h * kHeadDim;
#pragma unroll
    for (int dq = 0; dq < kQuads / 4; ++dq) {
      const int4 val = *reinterpret_cast<const int4*>(qrow + dq * 16);
      q_reg[dq * 4 + 0] = val.x; q_reg[dq * 4 + 1] = val.y;
      q_reg[dq * 4 + 2] = val.z; q_reg[dq * 4 + 3] = val.w;
    }
  } else {
#pragma unroll
    for (int dq = 0; dq < kQuads; ++dq) q_reg[dq] = 0;
  }

  float acc[kTM][kTD], row_m = kMaskedScore, row_l = 0.0f;
#pragma unroll
  for (int u = 0; u < kTM; ++u)
#pragma unroll
    for (int dd = 0; dd < kTD; ++dd) acc[u][dd] = 0.0f;

  auto stage_kv = [&](int dst, int kb0) {
#pragma unroll 1
    for (int i = tid; i < kBlockN * (kQuads / 4); i += kBlockM) {
      const int r = i / (kQuads / 4), c = i % (kQuads / 4), n = kb0 + r;
      int4 val = make_int4(0, 0, 0, 0);
      if (n < len)
        val = *reinterpret_cast<const int4*>(a.k + static_cast<std::int64_t>(col0 + n) * a.q_stride_n + h * kHeadDim +
                                             c * 16);
      *reinterpret_cast<int4*>(&k_buf[dst][r * kHeadDim + c * 16]) = val;
    }
    constexpr int kSlots = kHeadDim * kBlockN / 8;
#pragma unroll 1
    for (int u = 0; u < kSlots / kBlockM; ++u) {
      const int slot = tid + u * kBlockM;
      if (slot >= kSlots) continue;
      const int n_local = slot % kBlockN, dg = slot / kBlockN, n = kb0 + n_local;
      const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(a.v) +
                                 static_cast<std::int64_t>(col0 + n) * a.q_stride_n + h * kHeadDim + dg * 8;
      const int4 val = (n < len) ? *reinterpret_cast<const int4*>(src) : make_int4(0, 0, 0, 0);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
        v_buf[dst][(dg * 8 + jj) * kBlockN + n_local] = reinterpret_cast<const __half*>(&val)[jj];
    }
  };

  stage_kv(0, 0);
  __syncthreads();

#pragma unroll 1
  for (int kb = 0; kb < len; kb += kBlockN) {
    const int buf = (kb / kBlockN) & 1, nb = kb + kBlockN;
    if (nb < len) stage_kv(buf ^ 1, nb);

    float scr[kBlockN];
    bool keep_b[kBlockN];
    const bool ks_full = (kb + kBlockN) <= len;
    const int ks_grp0 = (col0 + kb) / kMinBlkK;
    const float ks_c = ks_full ? a.k_scale[static_cast<std::int64_t>(ks_grp0) * a.k_scale_stride_n + h] : 0.0f;
#pragma unroll
    for (int j0 = 0; j0 < kBlockN; j0 += 4) {
      int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll 1
      for (int dq = 0; dq < kQuads; dq += 4) {
        const int4 k0 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 0) * kHeadDim + dq * 4]);
        const int4 k1 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 1) * kHeadDim + dq * 4]);
        const int4 k2 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 2) * kHeadDim + dq * 4]);
        const int4 k3 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 3) * kHeadDim + dq * 4]);
        s0 = sdot4(q_reg[dq + 0], k0.x, s0); s0 = sdot4(q_reg[dq + 1], k0.y, s0);
        s0 = sdot4(q_reg[dq + 2], k0.z, s0); s0 = sdot4(q_reg[dq + 3], k0.w, s0);
        s1 = sdot4(q_reg[dq + 0], k1.x, s1); s1 = sdot4(q_reg[dq + 1], k1.y, s1);
        s1 = sdot4(q_reg[dq + 2], k1.z, s1); s1 = sdot4(q_reg[dq + 3], k1.w, s1);
        s2 = sdot4(q_reg[dq + 0], k2.x, s2); s2 = sdot4(q_reg[dq + 1], k2.y, s2);
        s2 = sdot4(q_reg[dq + 2], k2.z, s2); s2 = sdot4(q_reg[dq + 3], k2.w, s2);
        s3 = sdot4(q_reg[dq + 0], k3.x, s3); s3 = sdot4(q_reg[dq + 1], k3.y, s3);
        s3 = sdot4(q_reg[dq + 2], k3.z, s3); s3 = sdot4(q_reg[dq + 3], k3.w, s3);
      }
#pragma unroll
      for (int tj = 0; tj < 4; ++tj) {
        const int n = kb + j0 + tj;
        const bool keep = valid && (n < len);
        // A partial tile can end at a group past ks_grp0; read that key's own scale instead of the hoisted one.
        const float ksj = ks_full ? ks_c
                                  : a.k_scale[static_cast<std::int64_t>((col0 + n) / kMinBlkK) *
                                                  a.k_scale_stride_n + h];
        scr[j0 + tj] = keep ? static_cast<float>((tj == 0 ? s0 : tj == 1 ? s1 : tj == 2 ? s2 : s3) *
                                                 (qsv * ksj)) : 0.0f;
        keep_b[j0 + tj] = keep;
      }
    }

    float lm = kMaskedScore;
#pragma unroll
    for (int j = 0; j < kBlockN; ++j) if (keep_b[j]) lm = fmaxf(lm, scr[j]);
    const float gm = fmaxf(row_m, lm);
    const float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
    row_m = gm;
    row_l *= alpha;
    float ps = 0.0f;
#pragma unroll
    for (int j = 0; j < kBlockN; ++j) {
      const __half p16 = __float2half(keep_b[j] ? exp2f(scr[j] - row_m) : 0.0f);
      p_buf[j * kBlockM + tid] = p16;
      ps += __half2float(p16);
    }
    row_l += ps;
    alpha_buf[tid] = alpha;
    l_buf[tid] = row_l;

    __syncthreads();

    const int rg = tid % (kBlockM / kTM), dgo = tid / (kBlockM / kTM), r0 = rg * kTM;
#pragma unroll
    for (int u = 0; u < kTM; ++u)
#pragma unroll
      for (int dd = 0; dd < kTD; ++dd) acc[u][dd] *= alpha_buf[r0 + u];
#pragma unroll
    for (int kp = 0; kp < kBlockN / 2; ++kp) {
      const int k = kp * 2;
      unsigned pp[kTM];
#pragma unroll
      for (int up = 0; up < kTM / 2; ++up) {
        const unsigned pk0 = *reinterpret_cast<const unsigned*>(&p_buf[k * kBlockM + r0 + up * 2]);
        const unsigned pk1 = *reinterpret_cast<const unsigned*>(&p_buf[(k + 1) * kBlockM + r0 + up * 2]);
        pp[up * 2] = (pk0 & 0xffffu) | ((pk1 & 0xffffu) << 16);
        pp[up * 2 + 1] = (pk0 >> 16) | (pk1 & 0xffff0000u);
      }
#pragma unroll
      for (int dd = 0; dd < kTD; ++dd) {
        const int d = dgo * kTD + dd;
        const unsigned vv = *reinterpret_cast<const unsigned*>(&v_buf[buf][d * kBlockN + k]);
#pragma unroll
        for (int u = 0; u < kTM; ++u) acc[u][dd] = fdot2(pp[u], vv, acc[u][dd]);
      }
    }
    __syncthreads();
  }

  const int rg = tid % (kBlockM / kTM), dgo = tid / (kBlockM / kTM), r0 = rg * kTM;
#pragma unroll
  for (int u = 0; u < kTM; ++u) {
    const int qrow = static_cast<int>(blockIdx.x) * kBlockM + r0 + u;
    if (qrow >= len) continue;
    const float inv = l_buf[r0 + u] > 0.0f ? 1.0f / l_buf[r0 + u] : 0.0f;
#pragma unroll
    for (int dd = 0; dd < kTD; ++dd)
      a.out[static_cast<std::int64_t>(first + qrow) * a.ldo + h * kHeadDim + dgo * kTD + dd] =
          float_to_bf16(acc[u][dd] * inv);
  }
}

#else

// The dual-arch binary compiles this file for every AMD target. The launcher is host code shared by all of them and
// cannot test __GFX10__, so a non-RDNA2 code object still needs the symbol; callers must gate on device_is_rdna2().
__global__ void attn_kernel(const Gfx103xAttnArgs) {}

#endif  // __GFX10__

void fail(const char* what, hipError_t e) {
  throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

}  // namespace

void gfx103x_v_to_fp16(const sage_bf16* src, std::int64_t ld, const std::int32_t* cu, std::uint16_t* dst,
                       int tokens, int batch, hipStream_t stream) {
  const std::int64_t total = static_cast<std::int64_t>(tokens) * kHeadDim * hip_block::kHeads;
  const unsigned grid = static_cast<unsigned>((total + 255) / 256);
  v_to_fp16_kernel<<<grid, 256, 0, stream>>>(src, ld, cu, dst, tokens, batch);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) fail("gfx103x_v_to_fp16", e);
}

void gfx103x_attn(const Gfx103xAttnArgs& a, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((a.max_seqlen + kBlockM - 1) / kBlockM), static_cast<unsigned>(a.heads),
                  static_cast<unsigned>(a.batch));
  attn_kernel<<<grid, kBlockM, 0, stream>>>(a);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) fail("gfx103x_attn", e);
}

}  // namespace mi::hip

#else  // !ROCWMMA_I8_LEAF

#include <stdexcept>
#include "sage_attn_gfx103x.h"
namespace mi::hip {
void gfx103x_v_to_fp16(const sage_bf16*, std::int64_t, const std::int32_t*, std::uint16_t*, int, int, hipStream_t) {
  throw std::runtime_error("gfx103x V conversion requires an AMD build");
}
void gfx103x_attn(const Gfx103xAttnArgs&, hipStream_t) {
  throw std::runtime_error("gfx103x attention requires an AMD build");
}
}  // namespace mi::hip

#endif
