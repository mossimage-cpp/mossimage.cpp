// The INT8 ConvRot GEMM for the high tier: y [m, n] = bf16((x @ w^T) * xs * ws + bias), x [m, k] and w [n, k] int8,
// one per-row scale each. The int8 WMMA int32 leaves are in int8_leaf.h (raw gfx11 intrinsics).
//
// The kernel is the tiled, software-pipelined design used by the ROCm-optimised ComfyUI backend: an LDS-staged 128x128
// tile computed by several warps, with the next K tile's global reads prefetched into registers while the current tile's
// fragments are multiplied, and a two-deep register pipeline over the 16-byte WMMA K-steps. This removes the repeated
// global loads of the earlier one-wave 128x32 tile and is what closes most of the gap to ComfyUI's GEMM throughput.
#if defined(ROCWMMA_I8_LEAF)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "hip_block.h"
#include "int8_gemm.h"
#include "int8_leaf.h"
#include "runtime.h"

namespace hip_block {

using detail::add;
using detail::bf;
using detail::f;
using detail::rbf;

constexpr int kWave = 32;

struct Int8GemmArgs {
  const std::int8_t* x;
  std::int64_t ldx;
  const float* x_scale;
  const std::int8_t* w;
  const float* w_scale;
  const float* bias;
  bf16_bits* y;
  std::int64_t ldy;
  int m, n, k;
  const bf16_bits* lora_delta;  // optional [m, n] bf16 low-rank term, added before the bf16 rounding
  std::int64_t ldl;
  const bf16_bits* gate;        // optional [n] fp32 gate: y = residual + bf16(y)*gate, after the first bf16 rounding
  const bf16_bits* residual;    // optional [m, ldr], may alias y
  std::int64_t ldr;
  int m_block0 = 0;  // first M block of this chunk when the grid is split over M
};

#if defined(__GFX10__)

// RDNA2 has no WMMA. This is comfy-kitchen's gfx1035 VALU tile (Apache-2.0, Copyright (c) 2025 Comfy Org), adapted to
// this GEMM's scales/bias/LoRA/gate epilogue: a 128x128 block by 256 threads, each owning 16 rows x 4 columns, with
// V_DOT4_I32_I8. B is staged chunk-major: lane chunks of eight lanes read consecutive 16-byte addresses, which removes
// the LDS bank conflicts a 20-word row pitch would otherwise cause. The WMMA-tiled kernel below is still compiled so
// the host launcher's symbol exists in every object of the fat binary.
__global__ __launch_bounds__(256) void int8_gemm_valu(const Int8GemmArgs a, int x_blocks) {
  constexpr int TM = 16, TN = 4, BM = 128, BN = 128, BK = 64, THREADS = 256;
  constexpr int VPR = BK / 16;             // 16-byte K chunks per row per k-tile
  constexpr int A_SIZE = BM * BK, B_SIZE = BN * BK, NBUFFERS = 2;
  static_assert((BM / TM) * (BN / TN) == THREADS, "thread tile must cover the block");
  static_assert(NBUFFERS * (A_SIZE + B_SIZE) <= 64 * 1024, "LDS footprint would spill");
  __shared__ __align__(16) std::int8_t As[NBUFFERS][A_SIZE];
  __shared__ __align__(16) std::int8_t Bs[NBUFFERS][B_SIZE];

  const int tid = static_cast<int>(threadIdx.x);
  const int trow = tid / (BN / TN), tcol = tid % (BN / TN);
  const int orow = trow * TM, ocol = tcol * TN;
  // The launcher's grid is (N blocks, M blocks); keep the same grouped order as the WMMA kernel.
  const int blocks_n = gridDim.x, blocks_m = x_blocks;
  const int bid = blockIdx.y * blocks_n + blockIdx.x;
  constexpr int kGroupM = 4;
  const int per_group = kGroupM * blocks_n;
  const int group = bid / per_group, idx = bid - group * per_group;
  const int group_rows = std::min(kGroupM, blocks_m - group * kGroupM);
  const int m0 = (a.m_block0 + group * kGroupM + idx % group_rows) * BM, n0 = (idx / group_rows) * BN;

  auto load_tile = [&](int buf, int k0) {
    for (int i = tid; i < BM * VPR; i += THREADS) {
      const int r = i / VPR, c = i % VPR, gk = k0 + c * 16;
      int4 val = make_int4(0, 0, 0, 0);
      if (m0 + r < a.m && gk + 15 < a.k)
        val = *reinterpret_cast<const int4*>(a.x + static_cast<std::int64_t>(m0 + r) * a.ldx + gk);
      *reinterpret_cast<int4*>(&As[buf][r * BK + c * 16]) = val;
    }
    for (int i = tid; i < BN * VPR; i += THREADS) {
      const int r = i / VPR, c = i % VPR, gk = k0 + c * 16;
      int4 val = make_int4(0, 0, 0, 0);
      if (n0 + r < a.n && gk + 15 < a.k)
        val = *reinterpret_cast<const int4*>(a.w + static_cast<std::int64_t>(n0 + r) * a.k + gk);
      *reinterpret_cast<int4*>(&Bs[buf][(c * BN + r) * 16]) = val;
    }
  };

  int acc[TM][TN] = {};
  for (int p = 0; p < NBUFFERS - 1; ++p)
    if (p * BK < a.k) load_tile(p, p * BK);
  __syncthreads();

  using v4i = int __attribute__((ext_vector_type(4)));
  constexpr int PREFETCH = NBUFFERS - 1;
  int buf = 0;
  for (int k0 = 0; k0 < a.k; k0 += BK) {
    const int next = k0 + PREFETCH * BK;
    if (next < a.k) load_tile((buf + PREFETCH) % NBUFFERS, next);
    const std::int8_t* ac = &As[buf][0];
    const std::int8_t* bc = &Bs[buf][0];
    // Keep only one 16-byte K chunk's operands live at a time. Unrolling all four chunks keeps four times the v4i
    // operands in VGPRs, which costs an entire wave/SIMD.
#pragma unroll 1
    for (int c = 0; c < VPR; ++c) {
      v4i av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = *reinterpret_cast<const v4i*>(&ac[(orow + i) * BK + c * 16]);
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = *reinterpret_cast<const v4i*>(&bc[(c * BN + ocol + j) * 16]);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] = __builtin_amdgcn_sdot4(av[i][0], bv[j][0], acc[i][j], true);
          acc[i][j] = __builtin_amdgcn_sdot4(av[i][1], bv[j][1], acc[i][j], true);
          acc[i][j] = __builtin_amdgcn_sdot4(av[i][2], bv[j][2], acc[i][j], true);
          acc[i][j] = __builtin_amdgcn_sdot4(av[i][3], bv[j][3], acc[i][j], true);
        }
      }
    }
    __syncthreads();
    buf = (buf + 1) % NBUFFERS;
  }

  // Hoist the per-column operands: without this the compiler must assume the bf16 output store aliases w_scale/bias/
  // gate and re-issues every load for every row (comfy-kitchen gemm_wmma.h:346 has the same note).
  float wsc[TN], bi[TN];
  bf16_bits gt[TN];
  bool valid[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int col = n0 + ocol + j;
    valid[j] = col < a.n;
    wsc[j] = valid[j] ? a.w_scale[col] : 0.0f;
    bi[j] = (valid[j] && a.bias != nullptr) ? a.bias[col] : 0.0f;
    gt[j] = (valid[j] && a.gate != nullptr) ? a.gate[col] : 0;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + orow + i;
    if (row >= a.m) continue;
    const float xs = a.x_scale[row];
    const bf16_bits* ldrow = a.lora_delta != nullptr ? a.lora_delta + static_cast<std::int64_t>(row) * a.ldl : nullptr;
    bf16_bits out[TN] = {};
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      if (!valid[j]) continue;
      const int col = n0 + ocol + j;
      float v = static_cast<float>(acc[i][j]) * xs * wsc[j] + bi[j];
      if (ldrow != nullptr) v += f(ldrow[col]);
      const bf16_bits g = bf(v);
      if (a.gate != nullptr) {
        // bf(res + bf(bf(gemm) * gate)), the same rounding points as the bf16 GatedResidualOp.
        const float prod = rbf(f(g) * f(gt[j]));
        const float res = a.residual != nullptr ? f(a.residual[static_cast<std::int64_t>(row) * a.ldr + col]) : 0.0f;
        out[j] = bf(add(res, prod));
      } else {
        out[j] = g;
      }
    }
    // The thread tile covers four consecutive columns: two 32-bit stores instead of four 16-bit ones. A pair that
    // straddles the last column must stay scalar, otherwise its padding half would overwrite the next row (ldy = n).
    const std::size_t at = static_cast<std::size_t>(row) * a.ldy + n0 + ocol;
#pragma unroll
    for (int j = 0; j < TN; j += 2) {
      if (valid[j] && valid[j + 1])
        *reinterpret_cast<std::uint32_t*>(&a.y[at + j]) = static_cast<std::uint32_t>(out[j]) |
                                                          (static_cast<std::uint32_t>(out[j + 1]) << 16);
      else if (valid[j])
        a.y[at + j] = out[j];
      else if (valid[j + 1])
        a.y[at + j + 1] = out[j + 1];
    }
  }
}

#else

__global__ void int8_gemm_valu(const Int8GemmArgs, int) { __builtin_trap(); }

#endif

// Stage a ROWS x BKB byte tile from global into registers, then write it to LDS. One 16-byte chunk per lane per pass;
// out-of-range rows and the K tail are zero-filled so the WMMA reads never leave the tile.
template <int ROWS, int BKB, int THREADS, bool kTail>
struct TileStager {
  static constexpr int kPerRow = BKB / 16;
  static constexpr int kChunks = ROWS * kPerRow;
  static constexpr int kPerThread = kChunks / THREADS;
  static constexpr int kStride = BKB + 16;  // 16-byte pad keeps every LDS row aligned for one ds_read_b128
  static_assert(kChunks % THREADS == 0, "THREADS must divide the tile's 16-byte chunks");

  uint4 regs[kPerThread];

  __device__ __forceinline__ void load(const std::int8_t* src, int row0, int rows_total, std::int64_t ld,
                                       int kbyte0, int kbytes) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int c = tid * kPerThread + i;
      const int row = row0 + c / kPerRow;
      const int kbyte = kbyte0 + (c % kPerRow) * 16;
      if constexpr (kTail) {
        regs[i] = make_uint4(0, 0, 0, 0);
        if (row >= rows_total || kbyte >= kbytes) continue;
        const std::int8_t* p = src + static_cast<std::int64_t>(row) * ld + kbyte;
        if (kbyte + 16 <= kbytes) {
          regs[i] = *reinterpret_cast<const uint4*>(p);
        } else {
          // Fewer than 16 bytes left in this row: a vector load would cross into the next row (or past the buffer).
          unsigned char* d = reinterpret_cast<unsigned char*>(&regs[i]);
          const unsigned char* s = reinterpret_cast<const unsigned char*>(p);
#pragma unroll 1
          for (int b = 0; b < 16; ++b) d[b] = (kbyte + b < kbytes) ? s[b] : 0;
        }
      } else {
        regs[i] = (row < rows_total && kbyte < kbytes)
                      ? *reinterpret_cast<const uint4*>(src + static_cast<std::int64_t>(row) * ld + kbyte)
                      : make_uint4(0, 0, 0, 0);
      }
    }
  }

  __device__ __forceinline__ void store(std::int8_t* lds) const {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int c = tid * kPerThread + i;
      *reinterpret_cast<uint4*>(lds + (c / kPerRow) * kStride + (c % kPerRow) * 16) = regs[i];
    }
  }
};

// C[m, n] = epilogue(A[m, k] @ B[n, k]^T), B in its natural [N, K] row-major form. WARPS_M*WARPS_N waves of 32 threads
// tile a BM x BN output tile; each wave owns TM*TN 16x16 WMMA tiles. K walks in BKB-byte stages, each holding kSteps
// 16-byte WMMA K-steps. The next stage's global loads and the next K-step's fragments are issued ahead of the MMAs.
template <int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN, bool kTail>
__global__ __launch_bounds__(WARPS_M* WARPS_N* kWave) void int8_gemm_tiled(
    const Int8GemmArgs a, int x_blocks) {
  constexpr int kThreads = WARPS_M * WARPS_N * kWave;
  constexpr int kStride = BKB + 16;
  constexpr int kSteps = BKB / 16;
  static_assert(BM == WARPS_M * TM * 16 && BN == WARPS_N * TN * 16, "warp grid must tile the block");
  static_assert(TileStager<BM, BKB, kThreads, kTail>::kStride == kStride, "staging stride must match read stride");

  __shared__ __align__(16) std::int8_t As[BM * kStride];
  __shared__ __align__(16) std::int8_t Bs[BN * kStride];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int frag_r = lane & 15;  // gfx11: every lane holds the whole K-step of row frag_r

  // Grouped block order for L2 locality: consecutive blocks advance along M within a group of kGroupM block-rows,
  // so concurrently resident blocks share the same B columns. Grid x indexes N, y indexes M, as in the reference.
  constexpr int kGroupM = 4;
  const int blocks_n = gridDim.x;
  const int blocks_m = gridDim.y;
  const int bid = blockIdx.y * blocks_n + blockIdx.x;
  const int per_group = kGroupM * blocks_n;
  const int group = bid / per_group;
  const int idx = bid - group * per_group;
  const int group_rows = std::min(kGroupM, blocks_m - group * kGroupM);
  const int mb = group * kGroupM + idx % group_rows;
  const int nb = idx / group_rows;
  const int m0 = (a.m_block0 + mb) * BM, n0 = nb * BN;

  i8leaf::FragAcc acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) i8leaf::clear(acc[i][j]);
  }

  TileStager<BM, BKB, kThreads, kTail> sa, sb;
  // int8 elements are one byte: the row stride is ldx for A and k for B (the weight is [n, k] contiguous).
  sa.load(a.x, m0, a.m, a.ldx, 0, a.k);
  sb.load(a.w, n0, a.n, a.k, 0, a.k);
  sa.store(As);
  sb.store(Bs);
  __syncthreads();

  for (int k0 = 0; k0 < a.k; k0 += BKB) {
    const int knext = k0 + BKB;
    const bool has_next = knext < a.k;
    if (has_next) {
      sa.load(a.x, m0, a.m, a.ldx, knext, a.k);
      sb.load(a.w, n0, a.n, a.k, knext, a.k);
    }

    i8leaf::FragA af[2][TM];
    i8leaf::FragB bf[2][TN];
#if defined(__GFX10__)
    // RDNA2: each lane loads the eight A rows its accumulators need (a broadcast within each half-wave).
    const int half = (lane >> 4);
#pragma unroll
    for (int i = 0; i < TM; ++i) i8leaf::load_a(af[0][i], As, wm * (TM * 16) + i * 16, 0, kStride, half);
#else
#pragma unroll
    for (int i = 0; i < TM; ++i)
      af[0][i] = *reinterpret_cast<const i8leaf::v4i*>(As + (wm * (TM * 16) + i * 16 + frag_r) * kStride);
#endif
#pragma unroll
    for (int j = 0; j < TN; ++j)
      bf[0][j] = *reinterpret_cast<const i8leaf::v4i*>(Bs + (wn * (TN * 16) + j * 16 + frag_r) * kStride);

#pragma unroll
    for (int kk = 0; kk < kSteps; ++kk) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk + 1 < kSteps) {
        const int kb = (kk + 1) * 16;
#if defined(__GFX10__)
#pragma unroll
        for (int i = 0; i < TM; ++i) i8leaf::load_a(af[nxt][i], As, wm * (TM * 16) + i * 16, kb, kStride, half);
#else
#pragma unroll
        for (int i = 0; i < TM; ++i)
          af[nxt][i] = *reinterpret_cast<const i8leaf::v4i*>(As + (wm * (TM * 16) + i * 16 + frag_r) * kStride + kb);
#endif
#pragma unroll
        for (int j = 0; j < TN; ++j)
          bf[nxt][j] = *reinterpret_cast<const i8leaf::v4i*>(Bs + (wn * (TN * 16) + j * 16 + frag_r) * kStride + kb);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) i8leaf::mma(acc[i][j], af[cur][i], bf[cur][j]);
      }
    }

    if (has_next) {
      __syncthreads();  // all warps finished reading the current LDS tile
      sa.store(As);
      sb.store(Bs);
      __syncthreads();
    }
  }

  // Hoist the per-column operands: the bf16 output store could alias them, so without this the compiler re-issues
  // every scale/bias/gate load for each row (the same fix as comfy-kitchen's gemm_wmma.h epilogue).
  float wsc[TN], bi[TN];
  bf16_bits gt[TN];
  bool valid[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int c = n0 + wn * (TN * 16) + j * 16 + (lane & 15);
    valid[j] = c < a.n;
    wsc[j] = valid[j] ? a.w_scale[c] : 0.0f;
    bi[j] = (valid[j] && a.bias != nullptr) ? a.bias[c] : 0.0f;
    gt[j] = (valid[j] && a.gate != nullptr) ? a.gate[c] : 0;
  }
  // Each accumulator element e is output row 2*e + lane/16, column lane%16 on gfx11.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = m0 + wm * (TM * 16) + i * 16 + 2 * e + (lane >> 4);
      if (r >= a.m) continue;
      const float xs = a.x_scale[r];
      const bf16_bits* ldrow = a.lora_delta != nullptr ? a.lora_delta + static_cast<std::int64_t>(r) * a.ldl : nullptr;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        if (!valid[j]) continue;
        const int c = n0 + wn * (TN * 16) + j * 16 + (lane & 15);
        float v = static_cast<float>(acc[i][j][e]) * xs * wsc[j] + bi[j];
        if (ldrow != nullptr) v += f(ldrow[c]);
        const bf16_bits g = bf(v);  // the GEMM's own bf16 rounding, before the optional gate+residual epilogue
        const std::size_t at = static_cast<std::size_t>(r) * a.ldy + c;
        if (a.gate != nullptr) {
          // bf(x + bf(bf(gemm_out)*gate)): read the residual before storing to it (it may alias y).
          const float prod = rbf(f(g) * f(gt[j]));
          const float res = a.residual != nullptr ? f(a.residual[static_cast<std::int64_t>(r) * a.ldr + c]) : 0.0f;
          a.y[at] = bf(add(res, prod));
        } else {
          a.y[at] = g;
        }
      }
    }
  }
}

}  // namespace hip_block

namespace mi::hip {

// Launch the INT8 ConvRot GEMM: y [m, n] = bf16((x @ w^T) * xs * ws + bias). x [m, k] int8, w [n, k] int8.
void int8_gemm(hip_block::bf16_bits* y, std::int64_t ldy, const std::int8_t* x, std::int64_t ldx,
                       const float* x_scale, const std::int8_t* w, const float* w_scale, const float* bias, int m, int n,
                       int k, hipStream_t stream, const hip_block::bf16_bits* lora_delta, std::int64_t ldl,
                       const hip_block::bf16_bits* gate, const hip_block::bf16_bits* residual, std::int64_t ldr) {
  if (m == 0 || n == 0 || k == 0) return;
  hip_block::Int8GemmArgs a{x, ldx, x_scale, w, w_scale, bias, y, ldy, m, n, k, lora_delta, ldl,
                            gate, residual, ldr};
  // Chunk the grid's y dimension so a multi-megapixel image's token count cannot exceed the device's grid limit.
  constexpr int kChunk = 4096;
  const bool deep = k >= 4096;
  constexpr int bm = 128, bn = 128;
  const int gx = (m + bm - 1) / bm;
  const int gy = (n + bn - 1) / bn;
  for (int x0 = 0; x0 < gx; x0 += kChunk) {
    const int xn = std::min(kChunk, gx - x0);
    a.m_block0 = x0;
    dim3 grid(gy, xn, 1);  // x = N blocks, y = M blocks (matches the kernel's grouped ordering)
    // Deep K: the 128-thread 4x4 fragment grid has the highest matrix density and the LDS reads are amortised. Shallow
    // K: the 512-thread 2x2 grid yields more blocks to interleave across the GPU's WGPs. RDNA2 runs the VALU tile.
    if (device_is_rdna2()) {
      hip_block::int8_gemm_valu<<<grid, 256, 0, stream>>>(a, xn);
    } else if (deep) {
      // K is a multiple of 16 for every model GEMM; keep the original single-predicate staging path there and only
      // pay for a masked tail when a caller supplies an unusual K.
      if (k % 16 != 0)
        hip_block::int8_gemm_tiled<128, 128, 128, 2, 2, 4, 4, true><<<grid, 128, 0, stream>>>(a, xn);
      else
        hip_block::int8_gemm_tiled<128, 128, 128, 2, 2, 4, 4, false><<<grid, 128, 0, stream>>>(a, xn);
    } else {
      if (k % 16 != 0)
        hip_block::int8_gemm_tiled<128, 128, 64, 4, 4, 2, 2, true><<<grid, 512, 0, stream>>>(a, xn);
      else
        hip_block::int8_gemm_tiled<128, 128, 64, 4, 4, 2, 2, false><<<grid, 512, 0, stream>>>(a, xn);
    }
  }
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("hip_block::int8_gemm: ") + hipGetErrorString(e));
}

}  // namespace mi::hip

#else  // !ROCWMMA_I8_LEAF (the NVIDIA logic-check build has no gfx11 int8 WMMA)

#include <cstdint>
#include <stdexcept>

#include "hip_block.h"
#include "int8_gemm.h"

namespace mi::hip {

// Same signature as int8_gemm.h: a caller in the NVIDIA build must link even though the kernel cannot run there.
void int8_gemm(hip_block::bf16_bits* y, std::int64_t ldy, const std::int8_t* x, std::int64_t ldx,
               const float* x_scale, const std::int8_t* w, const float* w_scale, const float* bias, int m, int n, int k,
               hipStream_t stream, const hip_block::bf16_bits* lora_delta, std::int64_t ldl,
               const hip_block::bf16_bits* gate, const hip_block::bf16_bits* residual, std::int64_t ldr) {
  (void)y;
  (void)ldy;
  (void)x;
  (void)ldx;
  (void)x_scale;
  (void)w;
  (void)w_scale;
  (void)bias;
  (void)m;
  (void)n;
  (void)k;
  (void)stream;
  (void)lora_delta;
  (void)ldl;
  (void)gate;
  (void)residual;
  (void)ldr;
  throw std::runtime_error("the INT8 GEMM needs a gfx11 int8 WMMA build");
}

}  // namespace mi::hip

#endif
