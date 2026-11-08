// sm_75 VAE kernels: the reference decoder's convolutions and mid-block attention are TF32 tensor-core kernels (CUTLASS
// implicit GEMM with cp.async, and a TF32 m16n8k8 FlashAttention), neither of which exists on Turing. These fp32 SIMT
// replacements keep the reference precision (fp32 accumulate, fp32 softmax) at Turing speeds: the VAE is a small part
// of a generation, so plain fp32 is the right trade for a first working path.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "core/cuda_check.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::sm75 {
namespace {

// ---------------------------------------------------------------- conv2d (NHWC, KRSC filters, stride 1, pad R/2)
// One block: an 8x8 output tile of 32 output channels. Input channels are walked in groups of 8 through shared memory
// (the input patch and the weights they multiply), each thread accumulating 2 pixels x 16 channels.
constexpr int kTW = 8, kTH = 16, kTK = 32, kTC = 8;
constexpr int kPX = kTW + 2, kPY = kTH + 2;
constexpr int kConvThreads = 128;  // each thread computes two output pixels with the same weights

__global__ void conv2d_f32_sm75(const float* __restrict__ x, const float* __restrict__ w,
                                const float* __restrict__ bias, const float* __restrict__ residual,
                                float* __restrict__ y, int H, int W, int C, int K, int R, int kb_count) {
  __shared__ __align__(16) float xs[kPY * kPX * kTC];
  __shared__ __align__(16) float ws[kTK * 9 * kTC];
  const int n = blockIdx.z / kb_count;
  const int kb = blockIdx.z % kb_count;
  const int ox0 = blockIdx.x * kTW, oy0 = blockIdx.y * kTH;
  const int pad = (R - 1) / 2;
  const int tid = threadIdx.x;
  const int pixel = tid >> 1;      // 0..63; the second pixel is eight rows below
  const int cg = (tid & 1) * 16;   // first / second 16 output channels
  const int py = pixel / kTW, px = pixel % kTW;
  float4 acc[2][16];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
#pragma unroll
    for (int ch = 0; ch < 16; ++ch) acc[q][ch] = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  for (int c0 = 0; c0 < C; c0 += kTC) {
    for (int i = tid; i < kPY * kPX * kTC; i += kConvThreads) {
      const int cc = i % kTC, p = i / kTC;
      const int yy = p / kPX, xx = p % kPX;
      const int gy = oy0 - pad + yy, gx = ox0 - pad + xx, gc = c0 + cc;
      xs[i] = (gy >= 0 && gy < H && gx >= 0 && gx < W && gc < C)
                  ? x[((static_cast<std::int64_t>(n) * H + gy) * W + gx) * C + gc]
                  : 0.f;
    }
    for (int i = tid; i < kTK * 9 * kTC; i += kConvThreads) {
      const int cc = i % kTC, t = i / kTC, rs = t % 9, kk = t / 9;
      const int kch = kb * kTK + kk, gc = c0 + cc;
      // The tile holds 9 taps; a 1x1 filter (R = 1) has one, and reading the other 8 would run past the end of w.
      ws[i] = (kch < K && gc < C && rs < R * R)
                  ? w[(static_cast<std::int64_t>(kch) * R + rs / R) * R * C + (rs % R) * C + gc]
                  : 0.f;
    }
    __syncthreads();
    // Four input channels at a time: one float4 shared load per (channel, tap) feeds four FMAs into each of the two
    // pixels' float4 accumulators, so the weight tile is loaded for two output pixels at once.
    for (int cc4 = 0; cc4 < kTC; cc4 += 4) {
      if (c0 + cc4 + 3 < C) {
        for (int r = 0; r < R; ++r) {
          for (int s = 0; s < R; ++s) {
            const float4 xv0 = *reinterpret_cast<const float4*>(&xs[((py + r) * kPX + (px + s)) * kTC + cc4]);
            const float4 xv1 = *reinterpret_cast<const float4*>(&xs[((py + 8 + r) * kPX + (px + s)) * kTC + cc4]);
#pragma unroll
            for (int ch = 0; ch < 16; ++ch) {
              const float4 wv = *reinterpret_cast<const float4*>(&ws[((cg + ch) * 9 + r * R + s) * kTC + cc4]);
              acc[0][ch].x += xv0.x * wv.x;
              acc[0][ch].y += xv0.y * wv.y;
              acc[0][ch].z += xv0.z * wv.z;
              acc[0][ch].w += xv0.w * wv.w;
              acc[1][ch].x += xv1.x * wv.x;
              acc[1][ch].y += xv1.y * wv.y;
              acc[1][ch].z += xv1.z * wv.z;
              acc[1][ch].w += xv1.w * wv.w;
            }
          }
        }
      } else {
        for (int t = cc4; t < kTC && c0 + t < C; ++t) {
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < R; ++s) {
              const float xv0 = xs[((py + r) * kPX + (px + s)) * kTC + t];
              const float xv1 = xs[((py + 8 + r) * kPX + (px + s)) * kTC + t];
#pragma unroll
              for (int ch = 0; ch < 16; ++ch) {
                const float v = ws[((cg + ch) * 9 + r * R + s) * kTC + t];
                acc[0][ch].x += xv0 * v;
                acc[1][ch].x += xv1 * v;
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int ox = ox0 + px, oy = oy0 + py + q * 8;
    if (ox >= W || oy >= H) continue;
    float h[16];
#pragma unroll
    for (int ch = 0; ch < 16; ++ch) h[ch] = (acc[q][ch].x + acc[q][ch].y) + (acc[q][ch].z + acc[q][ch].w);
    const std::int64_t out_base = ((static_cast<std::int64_t>(n) * H + oy) * W + ox) * K;
    if ((K & 3) == 0 && kb * kTK + cg + 16 <= K) {
      const int kch = kb * kTK + cg;
#pragma unroll
      for (int ch = 0; ch < 16; ch += 4) {
        const float4 b = *reinterpret_cast<const float4*>(&bias[kch + ch]);
        float4 v = make_float4(h[ch] + b.x, h[ch + 1] + b.y, h[ch + 2] + b.z, h[ch + 3] + b.w);
        if (residual != nullptr) {
          const float4 r4 = *reinterpret_cast<const float4*>(&residual[out_base + kch + ch]);
          v.x += r4.x;
          v.y += r4.y;
          v.z += r4.z;
          v.w += r4.w;
        }
        *reinterpret_cast<float4*>(&y[out_base + kch + ch]) = v;
      }
    } else {
#pragma unroll
      for (int ch = 0; ch < 16; ++ch) {
        const int kch = kb * kTK + cg + ch;
        if (kch >= K) break;
        float v = h[ch] + bias[kch];
        if (residual != nullptr) v += residual[out_base + kch];
        y[out_base + kch] = v;
      }
    }
  }
}

// ---------------------------------------------------------------- mid-block attention (single head, fp32)
// One warp per row read K and V from global memory in full for every query row: at a 1024^2 latent (16384 tokens) that
// is 8 TB of reads and 5.4 s on an RTX 2080. This version keeps 32 query rows per block in registers (four per warp)
// and stages the K and V tiles once per block in shared memory, so every tile is read L/32 times instead of L times.
// D is a multiple of 32 (384 or 512); the online softmax keeps the same key order, so the output bits are unchanged.
constexpr int kAttnWarps = 8;
constexpr int kAttnRowsPerWarp = 4;
constexpr int kAttnRows = kAttnWarps * kAttnRowsPerWarp;  // 32 query rows per block
constexpr int kAttnTile = 8;                              // keys per staged tile

// LANES is D/32 at compile time, so the per-row register arrays keep their indices in registers; a runtime lane count
// would place them in local memory and cost an order of magnitude.
template <int LANES>
__global__ void attention_f32_sm75_t(const float* __restrict__ q, const float* __restrict__ k,
                                     const float* __restrict__ v, float* __restrict__ o, int L, int q_rows,
                                     int k_rows, int v_rows, std::int64_t q_batch, std::int64_t k_batch,
                                     std::int64_t v_batch, std::int64_t o_batch, float scale) {
  constexpr int kD = LANES * 32;
  extern __shared__ __align__(16) float smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tid = threadIdx.x;
  const int b = blockIdx.y;
  const int row0 = blockIdx.x * kAttnRows + warp * kAttnRowsPerWarp;
  float qreg[kAttnRowsPerWarp][LANES];
  float oacc[kAttnRowsPerWarp][LANES];
  float m[kAttnRowsPerWarp], s[kAttnRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kAttnRowsPerWarp; ++r) {
    const bool valid = row0 + r < L;
    const float* qrow = q + static_cast<std::int64_t>(b) * q_batch + static_cast<std::int64_t>(row0 + r) * q_rows;
    m[r] = -1e30f;
    s[r] = 0.f;
#pragma unroll
    for (int i = 0; i < LANES; ++i) {
      qreg[r][i] = valid ? qrow[lane + 32 * i] : 0.f;
      oacc[r][i] = 0.f;
    }
  }
  const float* kbase = k + static_cast<std::int64_t>(b) * k_batch;
  const float* vbase = v + static_cast<std::int64_t>(b) * v_batch;
  float* ktile = smem;
  float* vtile = smem + kAttnTile * kD;
  for (int j0 = 0; j0 < L; j0 += kAttnTile) {
    for (int idx = tid; idx < kAttnTile * kD; idx += kAttnWarps * 32) {
      const int jj = idx / kD, d = idx % kD;
      const int key = j0 + jj;
      const bool valid = key < L;
      ktile[idx] = valid ? kbase[static_cast<std::int64_t>(key) * k_rows + d] : 0.f;
      vtile[idx] = valid ? vbase[static_cast<std::int64_t>(key) * v_rows + d] : 0.f;
    }
    __syncthreads();
    const int jn = min(kAttnTile, L - j0);
    for (int jj = 0; jj < jn; ++jj) {
      const float* krow = ktile + jj * kD;
      const float* vrow = vtile + jj * kD;
#pragma unroll
      for (int r = 0; r < kAttnRowsPerWarp; ++r) {
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < LANES; ++i) dot += qreg[r][i] * krow[lane + 32 * i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        dot *= scale;
        const float mn = fmaxf(m[r], dot);
        const float alpha = (m[r] > -1e29f) ? expf(m[r] - mn) : 0.f;
        const float p = expf(dot - mn);
        s[r] = s[r] * alpha + p;
#pragma unroll
        for (int i = 0; i < LANES; ++i) oacc[r][i] = oacc[r][i] * alpha + p * vrow[lane + 32 * i];
        m[r] = mn;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < kAttnRowsPerWarp; ++r) {
    if (row0 + r >= L) continue;
    const float inv = 1.f / s[r];
    float* orow = o + static_cast<std::int64_t>(b) * o_batch + static_cast<std::int64_t>(row0 + r) * kD;
    for (int i = 0; i < LANES; ++i) orow[lane + 32 * i] = oacc[r][i] * inv;
  }
}
}  // namespace

void vae_conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
                cudaStream_t stream) {
  const int H = static_cast<int>(x.dim(1)), W = static_cast<int>(x.dim(2)), C = static_cast<int>(x.dim(3));
  const int K = static_cast<int>(w.dim(0)), R = static_cast<int>(w.dim(1));
  const int kb_count = (K + kTK - 1) / kTK;
  const dim3 grid(static_cast<unsigned>((W + kTW - 1) / kTW), static_cast<unsigned>((H + kTH - 1) / kTH),
                  static_cast<unsigned>(x.dim(0) * kb_count));
  conv2d_f32_sm75<<<grid, kConvThreads, 0, stream>>>(
      x.ptr<float>(), w.ptr<float>(), bias.ptr<float>(), residual != nullptr ? residual->ptr<float>() : nullptr,
      y.ptr<float>(), H, W, C, K, R, kb_count);
  MI_CUDA_CHECK_LAUNCH();
}


// ---------------------------------------------------------------- stride-2 downsample conv (VAE encoder)
// Same layout as conv2d_f32_sm75 (one block: an 8x8 output tile of 32 output channels, input channels in groups of 8),
// with the encoder's Downsample2D semantics: out[i,j] accumulates the 3x3 window at (2i, 2j) of the input, treating the
// input as zero outside [0,H) x [0,W) (the reference pads one row/column at the bottom/right, nothing at the top/left).
constexpr int kDownTW = 8, kDownTH = 8, kDownTC = 8;
constexpr int kDownPX = kDownTW * 2 + 2;  // 18
constexpr int kDownPY = kDownTH * 2 + 2;

__global__ void conv2d_down_f32_sm75(const float* __restrict__ x, const float* __restrict__ w,
                                     const float* __restrict__ bias, float* __restrict__ y, int H, int W, int C, int K,
                                     int P, int Q, int kb_count) {
  __shared__ float xs[kDownPY * kDownPX * kDownTC];
  __shared__ float ws[kTK * 9 * kDownTC];
  const int n = blockIdx.z / kb_count;
  const int kb = blockIdx.z % kb_count;
  const int ox0 = blockIdx.x * kDownTW, oy0 = blockIdx.y * kDownTH;
  const int tid = threadIdx.x;
  const int pixel = tid >> 1;
  const int cg = (tid & 1) * 16;
  const int py = pixel / kDownTW, px = pixel % kDownTW;
  float acc[16] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int c0 = 0; c0 < C; c0 += kDownTC) {
    for (int i = tid; i < kDownPY * kDownPX * kDownTC; i += kConvThreads) {
      const int cc = i % kDownTC, p = i / kDownTC;
      const int yy = p / kDownPX, xx = p % kDownPX;
      const int gy = oy0 * 2 + yy, gx = ox0 * 2 + xx, gc = c0 + cc;
      xs[i] = (gy < H && gx < W && gc < C) ? x[((static_cast<std::int64_t>(n) * H + gy) * W + gx) * C + gc] : 0.f;
    }
    for (int i = tid; i < kTK * 9 * kDownTC; i += kConvThreads) {
      const int cc = i % kDownTC, t = i / kDownTC, rs = t % 9, kk = t / 9;
      const int kch = kb * kTK + kk, gc = c0 + cc;
      ws[i] = (kch < K && gc < C) ? w[(static_cast<std::int64_t>(kch) * 3 + rs / 3) * 3 * C + (rs % 3) * C + gc] : 0.f;
    }
    __syncthreads();
    for (int cc = 0; cc < kDownTC; ++cc) {
      if (c0 + cc >= C) break;
      for (int r = 0; r < 3; ++r) {
        for (int s = 0; s < 3; ++s) {
          const float xv = xs[((py * 2 + r) * kDownPX + (px * 2 + s)) * kDownTC + cc];
#pragma unroll
          for (int ch = 0; ch < 16; ++ch) acc[ch] += xv * ws[((cg + ch) * 9 + r * 3 + s) * kDownTC + cc];
        }
      }
    }
    __syncthreads();
  }

  const int ox = ox0 + px, oy = oy0 + py;
  if (ox >= Q || oy >= P) return;
#pragma unroll
  for (int ch = 0; ch < 16; ++ch) {
    const int kch = kb * kTK + cg + ch;
    if (kch >= K) break;
    const std::int64_t idx = ((static_cast<std::int64_t>(n) * P + oy) * Q + ox) * K + kch;
    y[idx] = acc[ch] + bias[kch];
  }
}
void vae_conv2d_down(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor& y, cudaStream_t stream) {
  const int H = static_cast<int>(x.dim(1)), W = static_cast<int>(x.dim(2)), C = static_cast<int>(x.dim(3));
  const int K = static_cast<int>(w.dim(0));
  const int P = static_cast<int>(y.dim(1)), Q = static_cast<int>(y.dim(2));
  const int kb_count = (K + kTK - 1) / kTK;
  const dim3 grid(static_cast<unsigned>((Q + kDownTW - 1) / kDownTW), static_cast<unsigned>((P + kDownTH - 1) / kDownTH),
                  static_cast<unsigned>(x.dim(0) * kb_count));
  conv2d_down_f32_sm75<<<grid, kConvThreads, 0, stream>>>(x.ptr<float>(), w.ptr<float>(), bias.ptr<float>(),
                                                          y.ptr<float>(), H, W, C, K, P, Q, kb_count);
  MI_CUDA_CHECK_LAUNCH();
}
// Generic fallback (any D that is a multiple of 32): one warp per query row, the lane holds D/32 elements. Slow at long
// sequences, used only for head dimensions the register-tiled kernel does not instantiate (the checks' tiny shapes).
__global__ void attention_f32_sm75_dyn(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v, float* __restrict__ o, int L, int D, int q_rows,
                                       int k_rows, int v_rows, std::int64_t q_batch, std::int64_t k_batch,
                                       std::int64_t v_batch, std::int64_t o_batch, float scale) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 4 + warp;
  if (row >= L) return;
  const int nlanes = D / 32;
  const int b = blockIdx.y;
  const float* qrow = q + static_cast<std::int64_t>(b) * q_batch + static_cast<std::int64_t>(row) * q_rows;
  float qreg[16];
  float oacc[16];
  for (int i = 0; i < nlanes; ++i) {
    qreg[i] = qrow[lane + 32 * i];
    oacc[i] = 0.f;
  }
  float m = -1e30f, s = 0.f;
  for (int j = 0; j < L; ++j) {
    const float* krow = k + static_cast<std::int64_t>(b) * k_batch + static_cast<std::int64_t>(j) * k_rows;
    float dot = 0.f;
    for (int i = 0; i < nlanes; ++i) dot += qreg[i] * krow[lane + 32 * i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    dot *= scale;
    const float mn = fmaxf(m, dot);
    const float alpha = (m > -1e29f) ? expf(m - mn) : 0.f;
    const float p = expf(dot - mn);
    s = s * alpha + p;
    const float* vrow = v + static_cast<std::int64_t>(b) * v_batch + static_cast<std::int64_t>(j) * v_rows;
    for (int i = 0; i < nlanes; ++i) oacc[i] = oacc[i] * alpha + p * vrow[lane + 32 * i];
    m = mn;
  }
  const float inv = 1.f / s;
  float* orow = o + static_cast<std::int64_t>(b) * o_batch + static_cast<std::int64_t>(row) * D;
  for (int i = 0; i < nlanes; ++i) orow[lane + 32 * i] = oacc[i] * inv;
}

void vae_attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, float scale,
                        cudaStream_t stream) {
  const int L = static_cast<int>(q.dim(1)), D = static_cast<int>(q.dim(2));
  const int batch = static_cast<int>(q.dim(0));
  const dim3 grid(static_cast<unsigned>((L + kAttnRows - 1) / kAttnRows), static_cast<unsigned>(batch));
  const std::size_t smem = static_cast<std::size_t>(2) * kAttnTile * D * sizeof(float);
  const auto launch = [&](auto* kernel) {
    if (smem > 48 * 1024) {
      MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)));
    }
    kernel<<<grid, kAttnWarps * 32, smem, stream>>>(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(),
                                                    o.ptr<float>(), L, static_cast<int>(q.stride[1]),
                                                    static_cast<int>(k.stride[1]), static_cast<int>(v.stride[1]),
                                                    q.stride[0], k.stride[0], v.stride[0], o.stride[0], scale);
  };
  if (D == 384) {
    launch(&attention_f32_sm75_t<12>);
  } else if (D == 512) {
    launch(&attention_f32_sm75_t<16>);
  } else {
    const dim3 g(static_cast<unsigned>((L + 3) / 4), static_cast<unsigned>(batch));
    attention_f32_sm75_dyn<<<g, 4 * 32, 0, stream>>>(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), o.ptr<float>(),
                                                     L, D, static_cast<int>(q.stride[1]), static_cast<int>(k.stride[1]),
                                                     static_cast<int>(v.stride[1]), q.stride[0], k.stride[0],
                                                     v.stride[0], o.stride[0], scale);
  }
  MI_CUDA_CHECK_LAUNCH();
}

template <typename Out>
__global__ void kcrs_to_krsc_sm75(const float* __restrict__ src, Out* __restrict__ dst, int k, int c, int rs) {
  const std::int64_t total = static_cast<std::int64_t>(k) * c * rs;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int ci = static_cast<int>(i % c);
    const std::int64_t t = i / c;
    const int p = static_cast<int>(t % rs);
    const std::int64_t ki = t / rs;
    dst[i] = static_cast<Out>(src[(ki * c + ci) * rs + p]);
  }
}

void vae_weight_to_krsc(const Tensor& w_kcrs, const Tensor& w_krsc, cudaStream_t stream) {
  const int k = static_cast<int>(w_kcrs.dim(0)), c = static_cast<int>(w_kcrs.dim(1));
  const int rs = static_cast<int>(w_kcrs.dim(2) * w_kcrs.dim(3));
  const std::int64_t total = w_kcrs.numel();
  if (total == 0) return;
  const int blocks = static_cast<int>(std::min<std::int64_t>((total + 255) / 256, 4096));
  if (w_krsc.dtype == DType::kF32) {
    kcrs_to_krsc_sm75<float><<<blocks, 256, 0, stream>>>(w_kcrs.ptr<float>(), w_krsc.ptr<float>(), k, c, rs);
  } else if (w_krsc.dtype != DType::kBF16) {
    throw std::invalid_argument("sm75::vae_weight_to_krsc: destination must be f32 or bf16");
  } else {
    kcrs_to_krsc_sm75<__nv_bfloat16><<<blocks, 256, 0, stream>>>(w_kcrs.ptr<float>(), w_krsc.ptr<__nv_bfloat16>(), k,
                                                                 c, rs);
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::sm75
