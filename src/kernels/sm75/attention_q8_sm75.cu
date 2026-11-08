// 8-bit attention on sm_75 (`q8_fp16` / `q8_fp16_f32acc`): INT8 QK^T with Turing's m8n8k16 MMA and fp16 P·V with the
// existing m16n8k8 path. q and k are quantized per row (amax / 127) in a pre-pass; k is not mean-smoothed (the engine's
// q8 is lossy by definition, and the smoothing only buys INT8 range). v becomes fp16. The tiling, online softmax and
// P·V reuse the exact kernel's (attention_sm75.cu): 64 queries per CTA, 4 warps, 16 rows per warp, 32 keys per tile.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "core/cuda_check.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::sm75 {
namespace {

constexpr int kQ8D = 128;  // q8 attention is head_dim 128 only (the DiT's joint attention)
constexpr int kQ8BN = 32;
constexpr int kQ8Warps = 4;
constexpr int kQ8BM = kQ8Warps * 16;
constexpr int kQ8Stride = kQ8D + 16;  // int8 K tile row stride (4-byte aligned, pads the bank pattern)
constexpr int kQ8Chunks = kQ8D / 16;     // m8n8k16 in-dimension chunks (QK^T)
constexpr int kQ8PvChunks = kQ8D / 8;   // m16n8k8 in-dimension chunks (P·V)
constexpr float kLog2e = 1.4426950408889634f;

// m8n8k16 int8 -> int32 MMA (Turing). a = 4 int8 of one query row, b = 4 int8 of one key, d = 2 int32.
__device__ __forceinline__ void mma_m8n8k16(int& d0, int& d1, std::uint32_t a, std::uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
               : "+r"(d0), "+r"(d1)
               : "r"(a), "r"(b));
}

__device__ __forceinline__ void mma_m16n8k8(float (&d)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ std::uint32_t pack_half2(float lo, float hi) {
  union {
    __half2 h;
    std::uint32_t u;
  } c;
  c.h = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
  return c.u;
}

__device__ __forceinline__ std::uint32_t pack_bf162(float lo, float hi) {
  union {
    __nv_bfloat162 h;
    std::uint32_t u;
  } c;
  c.h = __floats2bfloat162_rn(lo, hi);
  return c.u;
}

__device__ __forceinline__ std::uint32_t load_u32(const void* p) {
  return *reinterpret_cast<const std::uint32_t*>(p);
}

__device__ __forceinline__ float block_amax(float x) {
  __shared__ float sm[kQ8Warps * 8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, off));
  if (lane == 0) sm[warp] = x;
  __syncthreads();
  float m = 0.f;
  for (int w = 0; w < 4; ++w) m = fmaxf(m, sm[w]);
  __syncthreads();
  return m;
}

// One CTA per (token, head): scale = amax / 127, then q = round(x / scale) clamped to [-127, 127].
__global__ void q8_quantize_rows(const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ y,
                                 float* __restrict__ scale, int rows, int heads, std::int64_t row_stride,
                                 std::int64_t head_stride) {
  const int idx = blockIdx.x;
  const int token = idx / heads, head = idx % heads;
  const __nv_bfloat16* src = x + static_cast<std::int64_t>(token) * row_stride + static_cast<std::int64_t>(head) * head_stride;
  float amax = 0.f;
  for (int i = threadIdx.x; i < kQ8D; i += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
  amax = block_amax(amax);
  const float s = amax > 0.f ? amax / 127.f : 1.f;
  if (threadIdx.x == 0) scale[idx] = s;
  const float inv = 1.f / s;
  std::int8_t* dst = y + (static_cast<std::int64_t>(token) * heads + head) * kQ8D;
  for (int i = threadIdx.x; i < kQ8D; i += blockDim.x) {
    int v = static_cast<int>(__float2int_rn(__bfloat162float(src[i]) * inv));
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    dst[i] = static_cast<std::int8_t>(v);
  }
  (void)rows;
}

// One CTA per (token, head): v bf16 -> fp16.
__global__ void q8_convert_v(const __nv_bfloat16* __restrict__ v, __half* __restrict__ out, std::int64_t row_stride,
                             std::int64_t head_stride, int heads) {
  const int idx = blockIdx.x;
  const int token = idx / heads, head = idx % heads;
  const __nv_bfloat16* src = v + static_cast<std::int64_t>(token) * row_stride + static_cast<std::int64_t>(head) * head_stride;
  __half* dst = out + (static_cast<std::int64_t>(token) * heads + head) * kQ8D;
  for (int i = threadIdx.x; i < kQ8D; i += blockDim.x) dst[i] = __float2half_rn(__bfloat162float(src[i]));
}

struct Q8Params {
  const std::int8_t* q = nullptr;
  const std::int8_t* k = nullptr;
  const __half* v = nullptr;
  const float* q_scale = nullptr;
  const float* k_scale = nullptr;
  __nv_bfloat16* o = nullptr;
  const int* cu_q = nullptr;
  const int* cu_k = nullptr;
  int heads = 0, kv_heads = 0, total_q = 0;
  std::int64_t q_row = 0, q_head = 0, k_row = 0, k_head = 0, v_row = 0, v_head = 0, o_row = 0, o_head = 0;
  float scale = 1.f;
};

__global__ void attention_q8_kernel(const Q8Params p) {
  __shared__ __align__(16) std::int8_t ks[kQ8BN * kQ8Stride];
  __shared__ __align__(16) __half vt[kQ8D * (kQ8BN + 8)];
  __shared__ float kscale[kQ8BN];
  const int batch = blockIdx.y, head = blockIdx.z;
  const int q_start = p.cu_q[batch], q_end = p.cu_q[batch + 1];
  const int k_start = p.cu_k[batch], k_end = p.cu_k[batch + 1];
  const int len_q = q_end - q_start, len_k = k_end - k_start;
  const int row_base = static_cast<int>(blockIdx.x) * kQ8BM;
  if (row_base >= len_q) return;
  const int rows = min(kQ8BM, len_q - row_base);
  const int kv_head = head / (p.heads / p.kv_heads);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int group = lane >> 2, tig = lane & 3;
  const int r0 = warp * 16 + group, r1 = r0 + 8;

  // Q fragments: 4 int8 of one query row per chunk; two row groups per lane-pair.
  std::uint32_t qf[2][kQ8Chunks];
  const std::int64_t q_head_off = static_cast<std::int64_t>(head) * p.q_head;
  const auto load_q = [&](int row, int chunk) -> std::uint32_t {
    if (row >= rows) return 0u;
    return load_u32(p.q + static_cast<std::int64_t>(q_start + row_base + row) * p.q_row + q_head_off + chunk * 16 + tig * 4);
  };
#pragma unroll
  for (int i = 0; i < kQ8Chunks; ++i) {
    qf[0][i] = load_q(r0, i);
    qf[1][i] = load_q(r1, i);
  }
  const float sq0 = r0 < rows ? p.q_scale[(q_start + row_base + r0) * p.heads + head] : 1.f;
  const float sq1 = r1 < rows ? p.q_scale[(q_start + row_base + r1) * p.heads + head] : 1.f;

  float o[kQ8PvChunks][4];
#pragma unroll
  for (int i = 0; i < kQ8PvChunks; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

  for (int j0 = 0; j0 < len_k; j0 += kQ8BN) {
    const int n = min(kQ8BN, len_k - j0);
    for (int idx = tid; idx < kQ8BN * kQ8D; idx += kQ8Warps * 32) {
      const int j = idx / kQ8D, d = idx % kQ8D;
      const int key = j0 + j;
      const bool valid = key < len_k;
      ks[j * kQ8Stride + d] =
          valid ? p.k[static_cast<std::int64_t>(k_start + key) * p.k_row + static_cast<std::int64_t>(kv_head) * p.k_head + d] : 0;
      vt[d * (kQ8BN + 8) + j] =
          valid ? p.v[static_cast<std::int64_t>(k_start + key) * p.v_row + static_cast<std::int64_t>(kv_head) * p.v_head + d]
                : __float2half_rn(0.f);
    }
    for (int idx = tid; idx < kQ8BN; idx += kQ8Warps * 32) {
      kscale[idx] = j0 + idx < len_k ? p.k_scale[(k_start + j0 + idx) * p.kv_heads + kv_head] : 0.f;
    }
    __syncthreads();
    constexpr int kSubtiles = kQ8BN / 8;
    float s[kSubtiles][4];
    const int jn = (n + 7) / 8;
#pragma unroll
    for (int t = 0; t < kSubtiles; ++t) {
      int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
      const int key = t * 8 + group;
#pragma unroll
      for (int i = 0; i < kQ8Chunks; ++i) {
        const std::uint32_t kb = load_u32(&ks[key * kQ8Stride + i * 16 + tig * 4]);
        mma_m8n8k16(a0, a1, qf[0][i], kb);
        mma_m8n8k16(b0, b1, qf[1][i], kb);
      }
      const float sk0 = kscale[t * 8 + 2 * tig] * p.scale;
      const float sk1 = kscale[t * 8 + 2 * tig + 1] * p.scale;
      s[t][0] = t < jn ? static_cast<float>(a0) * sq0 * sk0 : -1e30f;
      s[t][1] = t < jn ? static_cast<float>(a1) * sq0 * sk1 : -1e30f;
      s[t][2] = t < jn ? static_cast<float>(b0) * sq1 * sk0 : -1e30f;
      s[t][3] = t < jn ? static_cast<float>(b1) * sq1 * sk1 : -1e30f;
    }
    float tile_max0 = -1e30f, tile_max1 = -1e30f;
#pragma unroll
    for (int t = 0; t < kSubtiles; ++t) {
      const int key0 = j0 + t * 8 + tig * 2;
      for (int e = 0; e < 4; ++e) {
        const int key = key0 + (e & 1);
        if (key >= len_k) s[t][e] = -1e30f;
      }
      tile_max0 = fmaxf(tile_max0, fmaxf(s[t][0], s[t][1]));
      tile_max1 = fmaxf(tile_max1, fmaxf(s[t][2], s[t][3]));
    }
    tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffffu, tile_max0, 1));
    tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffffu, tile_max0, 2));
    tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffffu, tile_max1, 1));
    tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffffu, tile_max1, 2));
    const float m_new0 = fmaxf(m0, tile_max0), m_new1 = fmaxf(m1, tile_max1);
    const float alpha0 = (m0 <= -1e29f) ? 0.f : exp2f((m0 - m_new0) * kLog2e);
    const float alpha1 = (m1 <= -1e29f) ? 0.f : exp2f((m1 - m_new1) * kLog2e);
    const auto prob = [](float sv, float mv) { return sv <= -1e29f ? 0.f : exp2f((sv - mv) * kLog2e); };
    float tile_sum0 = 0.f, tile_sum1 = 0.f;
#pragma unroll
    for (int t = 0; t < kSubtiles; ++t) {
      const float p0 = prob(s[t][0], m_new0), p1 = prob(s[t][1], m_new0);
      const float p2 = prob(s[t][2], m_new1), p3 = prob(s[t][3], m_new1);
      tile_sum0 += p0 + p1;
      tile_sum1 += p2 + p3;
      s[t][0] = p0;
      s[t][1] = p1;
      s[t][2] = p2;
      s[t][3] = p3;
    }
    tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 1);
    tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 2);
    tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 1);
    tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 2);
    l0 = l0 * alpha0 + tile_sum0;
    l1 = l1 * alpha1 + tile_sum1;
    m0 = m_new0;
    m1 = m_new1;
#pragma unroll
    for (int i = 0; i < kQ8PvChunks; ++i) {
      o[i][0] *= alpha0;
      o[i][1] *= alpha0;
      o[i][2] *= alpha1;
      o[i][3] *= alpha1;
    }
#pragma unroll
    for (int t = 0; t < kSubtiles; ++t) {
      const std::uint32_t pa0 = pack_half2(s[t][0], s[t][1]);
      const std::uint32_t pa1 = pack_half2(s[t][2], s[t][3]);
#pragma unroll
      for (int i = 0; i < kQ8PvChunks; ++i) {
        const std::uint32_t b = load_u32(&vt[(i * 8 + group) * (kQ8BN + 8) + t * 8 + tig * 2]);
        mma_m16n8k8(o[i], pa0, pa1, b);
      }
    }
    __syncthreads();
  }

  const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f;
  const float inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
  if (r0 < rows) {
#pragma unroll
    for (int i = 0; i < kQ8PvChunks; ++i) {
      const std::int64_t idx = static_cast<std::int64_t>(q_start + row_base + r0) * p.o_row +
                               static_cast<std::int64_t>(head) * p.o_head + i * 8 + tig * 2;
      *reinterpret_cast<__nv_bfloat162*>(&p.o[idx]) = __floats2bfloat162_rn(o[i][0] * inv0, o[i][1] * inv0);
    }
  }
  if (r1 < rows) {
#pragma unroll
    for (int i = 0; i < kQ8PvChunks; ++i) {
      const std::int64_t idx = static_cast<std::int64_t>(q_start + row_base + r1) * p.o_row +
                               static_cast<std::int64_t>(head) * p.o_head + i * 8 + tig * 2;
      *reinterpret_cast<__nv_bfloat162*>(&p.o[idx]) = __floats2bfloat162_rn(o[i][2] * inv1, o[i][3] * inv1);
    }
  }
}

}  // namespace

std::size_t q8_workspace_bytes(std::int64_t total_q, std::int64_t total_k, int num_heads, int num_kv_heads) {
  const auto pad = [](std::size_t b) { return (b + 255u) & ~std::size_t{255}; };
  const std::size_t q8 = static_cast<std::size_t>(total_q * num_heads * kQ8D);
  const std::size_t k8 = static_cast<std::size_t>(total_k * num_kv_heads * kQ8D);
  const std::size_t v16 = k8 * 2;
  const std::size_t qs = static_cast<std::size_t>(total_q * num_heads) * sizeof(float);
  const std::size_t ks = static_cast<std::size_t>(total_k * num_kv_heads) * sizeof(float);
  return pad(q8) + pad(k8) + pad(v16) + pad(qs) + pad(ks);
}

void q8_flash_fwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& cu_seqlens_q,
                  const Tensor& cu_seqlens_k, int max_seqlen, float scale, void* workspace, std::size_t workspace_bytes,
                  cudaStream_t stream) {
  const std::int64_t total_q = q.dim(0);
  if (total_q == 0) return;
  const std::int64_t total_k = k.dim(0);
  const int heads = static_cast<int>(q.dim(1));
  const int kv_heads = static_cast<int>(k.dim(1));
  const int batch = static_cast<int>(cu_seqlens_q.dim(0) - 1);
  const auto pad = [](std::size_t b) { return (b + 255u) & ~std::size_t{255}; };
  std::size_t off = 0;
  std::int8_t* q_i8 = static_cast<std::int8_t*>(workspace) + off;
  off += pad(static_cast<std::size_t>(total_q * heads * kQ8D));
  std::int8_t* k_i8 = static_cast<std::int8_t*>(workspace) + off;
  off += pad(static_cast<std::size_t>(total_k * kv_heads * kQ8D));
  __half* v16 = reinterpret_cast<__half*>(static_cast<std::int8_t*>(workspace) + off);
  off += pad(static_cast<std::size_t>(total_k * kv_heads * kQ8D * 2));
  float* q_scale = reinterpret_cast<float*>(static_cast<std::int8_t*>(workspace) + off);
  off += pad(static_cast<std::size_t>(total_q * heads * sizeof(float)));
  float* k_scale = reinterpret_cast<float*>(static_cast<std::int8_t*>(workspace) + off);
  if (workspace_bytes < q8_workspace_bytes(total_q, total_k, heads, kv_heads)) {
    throw std::invalid_argument("sm75::q8_flash_fwd: workspace too small");
  }
  const __nv_bfloat16* qb = static_cast<const __nv_bfloat16*>(q.data);
  const __nv_bfloat16* kb = static_cast<const __nv_bfloat16*>(k.data);
  const __nv_bfloat16* vb = static_cast<const __nv_bfloat16*>(v.data);
  q8_quantize_rows<<<static_cast<int>(total_q * heads), 128, 0, stream>>>(qb, q_i8, q_scale, static_cast<int>(total_q), heads,
                                                                        q.stride[0], q.stride[1]);
  MI_CUDA_CHECK_LAUNCH();
  q8_quantize_rows<<<static_cast<int>(total_k * kv_heads), 128, 0, stream>>>(kb, k_i8, k_scale, static_cast<int>(total_k),
                                                                           kv_heads, k.stride[0], k.stride[1]);
  MI_CUDA_CHECK_LAUNCH();
  q8_convert_v<<<static_cast<int>(total_k * kv_heads), 128, 0, stream>>>(vb, v16, v.stride[0], v.stride[1], kv_heads);
  MI_CUDA_CHECK_LAUNCH();
  Q8Params p;
  p.q = q_i8;
  p.k = k_i8;
  p.v = v16;
  p.q_scale = q_scale;
  p.k_scale = k_scale;
  p.o = static_cast<__nv_bfloat16*>(out.data);
  p.cu_q = cu_seqlens_q.ptr<int>();
  p.cu_k = cu_seqlens_k.ptr<int>();
  p.heads = heads;
  p.kv_heads = kv_heads;
  p.total_q = static_cast<int>(total_q);
  p.q_row = static_cast<std::int64_t>(heads) * kQ8D;
  p.q_head = kQ8D;
  p.k_row = static_cast<std::int64_t>(kv_heads) * kQ8D;
  p.k_head = kQ8D;
  p.v_row = static_cast<std::int64_t>(kv_heads) * kQ8D;
  p.v_head = kQ8D;
  p.o_row = out.stride[0];
  p.o_head = out.stride[1];
  p.scale = scale;
  const dim3 grid(static_cast<unsigned>((max_seqlen + kQ8BM - 1) / kQ8BM), static_cast<unsigned>(batch),
                  static_cast<unsigned>(heads));
  attention_q8_kernel<<<grid, kQ8Warps * 32, 0, stream>>>(p);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::sm75
