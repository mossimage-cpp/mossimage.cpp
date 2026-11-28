// The swappable leaf of every GEMM and of attention: a 16x16x16 bf16 multiply with fp32 accumulators, per wave.
//
//   HIP_BLOCK_LEAF_ROCWMMA on gfx11/gfx12                 rocWMMA fragments, v_wmma_f32_16x16x16_bf16
//   HIP_BLOCK_LEAF_ROCWMMA on gfx10 (RDNA2)                v_dot2_f32_f16 pairs, bf16 operands converted to fp16
//   otherwise (NVIDIA verify build)                       scalar: the exact bf16*bf16 product in fp32, then add, ascending K
//
// RDNA2 has neither matrix cores nor bf16 instructions: the only hardware multiply-add for these operands is the fp16
// dot product. bf16 values are converted to fp16 on load (exact while inside fp16's range, which covers the model's
// activations and weights), and V_DOT2_F32_F16 accumulates the products in fp32. One wave still covers one 16x16x16
// fragment; each thread owns two rows by four columns of the accumulator, so it holds 2x16 A values and 4x16 B values
// and issues 2*4*8 dot2 instructions per K step.
//
// rocWMMA does not compile under nvcc, and its fragments' lane layouts are private and differ between RDNA3 and RDNA4.
// So nothing outside this file touches a fragment's elements: an accumulator's values reach the kernels only through
// store_acc, into memory.
//
// Every call is collective: all 32 threads of the wave make it, with the same (thread-independent) arguments and under
// the same control flow. p may point to global or __shared__ memory. A caller reading what store_acc wrote to
// __shared__ memory from another thread must call barrier() first.
// The kernels depend on lockstep: they reuse a __shared__ buffer (the zero-padded fragments, s_mem, o_mem) right after a
// collective read of it, with a barrier before the next write's readers but none between the read and that write. That
// is safe only because the 32 threads are one wave, or one warp that has not diverged at that point. A workgroup of
// several waves or warps would need a barrier after every such read.
//   load_a       A(m, k) = p[m * ld + k]
//   load_b       B(k, n) = p[n * ld + k]   (a [N, K] row-major weight, or K's tokens, one column per row of memory)
//   load_b_rows  B(k, n) = p[k * ld + n]   (V's tokens: K along the rows of memory)
//   mma          c += a * b
//   store_acc    p[m * ld + n] = c(m, n)
//   load_acc     c(m, n) = p[m * ld + n]
#pragma once

#include <cstdint>

#include "device_math.h"

#if defined(HIP_BLOCK_LEAF_ROCWMMA) && !defined(__GFX10__)

#include <rocwmma/rocwmma.hpp>

namespace hip_block::leaf {

inline constexpr const char* kName =
    "rocWMMA bf16 (reference block GEMMs); native gfx11 WMMA int8 (DiT GEMM/attention) + bf16 (TE/LoRA GEMMs)";

using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

// One wave per workgroup, so the lanes are in lockstep: a barrier only has to order this wave's own __shared__ accesses.
// __syncthreads would add a cache invalidate at every call, throwing away the operands in the wave's L0.
__device__ __forceinline__ void barrier() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront"); }
// barrier() between this wave's writes of a zero-padded tail copy and the fragment load of it. That load can be a FLAT
// load: the copy and the operand it stands in for merge into one generic pointer. Nothing documented orders a FLAT read of
// __shared__ memory after the wave's DS writes, and barrier() emits no wait, so this one waits for the DS writes too.
__device__ __forceinline__ void pad_barrier() {
  barrier();
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
}

__device__ __forceinline__ void clear(FragAcc& c) { rocwmma::fill_fragment(c, 0.0f); }
// kFp16 is only used by the gfx1035 single-piece VAE path; the other leaves keep their bf16 interface.
template <bool kFp16 = false>
__device__ __forceinline__ void load_a(FragA& a, const std::uint16_t* p, int ld) {
  rocwmma::load_matrix_sync(a, reinterpret_cast<const rocwmma::bfloat16_t*>(p), static_cast<std::uint32_t>(ld));
}
template <bool kFp16 = false>
__device__ __forceinline__ void load_b(FragB& b, const std::uint16_t* p, int ld) {
  rocwmma::load_matrix_sync(b, reinterpret_cast<const rocwmma::bfloat16_t*>(p), static_cast<std::uint32_t>(ld));
}
template <bool kFp16 = false>
__device__ __forceinline__ void load_b_rows(FragB& b, const std::uint16_t* p, int ld) {
  rocwmma::load_matrix_sync(b, reinterpret_cast<const rocwmma::bfloat16_t*>(p), static_cast<std::uint32_t>(ld),
                            rocwmma::mem_row_major);
}
__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) { rocwmma::mma_sync(c, a, b, c); }
__device__ __forceinline__ void store_acc(float* p, const FragAcc& c, int ld) {
  rocwmma::store_matrix_sync(p, c, static_cast<std::uint32_t>(ld), rocwmma::mem_row_major);
}
__device__ __forceinline__ void load_acc(FragAcc& c, const float* p, int ld) {
  rocwmma::load_matrix_sync(c, p, static_cast<std::uint32_t>(ld), rocwmma::mem_row_major);
}

}  // namespace hip_block::leaf

#elif defined(HIP_BLOCK_LEAF_ROCWMMA) && defined(__GFX10__)

// RDNA2 (gfx1030..gfx1036): no WMMA and no bf16 dot. `v_dot2_f32_f16` is the only hardware multiply-accumulate for
// these operands; its operands are fp16 pairs and its accumulator is fp32. Conversions happen in the fragment loads,
// so every caller keeps the same bf16-bits interface as the rocWMMA and scalar leaves.
#include <hip/hip_fp16.h>

namespace hip_block::leaf {

inline constexpr const char* kName = "RDNA2 v_dot2_f32_f16 (bf16 to fp16), emulated 16x16x16 leaf";

using v2h = _Float16 __attribute__((ext_vector_type(2)));

// Thread t of the wave owns rows 2*(t/4), 2*(t/4)+1 and columns 4*(t%4) .. +3 of a 16x16 accumulator. That shares each
// A row across four columns and each B column across two rows, which keeps the fragments small enough to stay in
// registers at the tile sizes the kernels use.
__device__ __forceinline__ int row_of_thread() { return ((static_cast<int>(threadIdx.x) & 31) >> 2) * 2; }
__device__ __forceinline__ int col_of_thread() { return ((static_cast<int>(threadIdx.x) & 31) & 3) * 4; }

// bf16 -> fp32 is a left shift of the bits. Clamping guards the (unexpected) value outside fp16's range against an
// infinity that would turn a later dot product into NaN.
// kFp16 reads the operand as fp16 bits directly (the VAE's single-piece path); the default keeps the bf16 interface.
template <bool kFp16 = false>
__device__ __forceinline__ v2h cvt2(std::uint16_t b0, std::uint16_t b1) {
  if constexpr (kFp16) {
    v2h r;
    auto* h = reinterpret_cast<__half*>(&r);
    h[0] = *reinterpret_cast<const __half*>(&b0);
    h[1] = *reinterpret_cast<const __half*>(&b1);
    return r;
  } else {
    const float f0 = detail::f(b0), f1 = detail::f(b1);
    return {__float2half_rn(fminf(fmaxf(f0, -65504.0f), 65504.0f)),
            __float2half_rn(fminf(fmaxf(f1, -65504.0f), 65504.0f))};
  }
}

struct FragA {
  v2h v[2][8];  // [owned row][k pair]
};
struct FragB {
  v2h v[4][8];  // [owned column][k pair]
};
struct FragAcc {
  float v[2][4];
};

__device__ __forceinline__ void barrier() { __syncthreads(); }
__device__ __forceinline__ void pad_barrier() { barrier(); }

__device__ __forceinline__ void clear(FragAcc& c) {
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 2; ++i) {
    HIP_BLOCK_UNROLL
    for (int j = 0; j < 4; ++j) c.v[i][j] = 0.0f;
  }
}

// A(m, k) = p[m * ld + k], row-major.
template <bool kFp16 = false>
__device__ __forceinline__ void load_a(FragA& a, const std::uint16_t* p, int ld) {
  const int r0 = row_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 2; ++i) {
    const std::uint16_t* row = p + (r0 + i) * ld;
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 8; ++k) a.v[i][k] = cvt2<kFp16>(row[2 * k], row[2 * k + 1]);
  }
}

// B(k, n) = p[n * ld + k]: a weight row, K along the row.
template <bool kFp16 = false>
__device__ __forceinline__ void load_b(FragB& b, const std::uint16_t* p, int ld) {
  const int c0 = col_of_thread();
  HIP_BLOCK_UNROLL
  for (int j = 0; j < 4; ++j) {
    const std::uint16_t* col = p + (c0 + j) * ld;
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 8; ++k) b.v[j][k] = cvt2<kFp16>(col[2 * k], col[2 * k + 1]);
  }
}

// B(k, n) = p[k * ld + n]: V's tokens, one column of memory per K row.
template <bool kFp16 = false>
__device__ __forceinline__ void load_b_rows(FragB& b, const std::uint16_t* p, int ld) {
  const int c0 = col_of_thread();
  HIP_BLOCK_UNROLL
  for (int j = 0; j < 4; ++j) {
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 8; ++k) b.v[j][k] = cvt2<kFp16>(p[(2 * k) * ld + c0 + j], p[(2 * k + 1) * ld + c0 + j]);
  }
}

__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) {
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 2; ++i) {
    HIP_BLOCK_UNROLL
    for (int j = 0; j < 4; ++j) {
      float acc = c.v[i][j];
      HIP_BLOCK_UNROLL
      for (int k = 0; k < 8; ++k) acc = __builtin_amdgcn_fdot2(a.v[i][k], b.v[j][k], acc, true);
      c.v[i][j] = acc;
    }
  }
}

__device__ __forceinline__ void store_acc(float* p, const FragAcc& c, int ld) {
  float* row = p + row_of_thread() * ld + col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 2; ++i) {
    HIP_BLOCK_UNROLL
    for (int j = 0; j < 4; ++j) row[i * ld + j] = c.v[i][j];
  }
}

__device__ __forceinline__ void load_acc(FragAcc& c, const float* p, int ld) {
  const float* row = p + row_of_thread() * ld + col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 2; ++i) {
    HIP_BLOCK_UNROLL
    for (int j = 0; j < 4; ++j) c.v[i][j] = row[i * ld + j];
  }
}

}  // namespace hip_block::leaf

#else

namespace hip_block::leaf {

inline constexpr const char* kName = "scalar (logic check, not WMMA)";

// Thread t of the wave owns row t / 2 and columns (t & 1) * 8 .. + 8 of the accumulator: the elements of each row are
// computed from that row of A and those columns of B, copied into its registers as a fragment load would.
__device__ __forceinline__ int row_of_thread() { return static_cast<int>(threadIdx.x) / 2; }
__device__ __forceinline__ int col_of_thread() { return (static_cast<int>(threadIdx.x) & 1) * 8; }

struct FragA {
  float v[16];
};
struct FragB {
  float v[8][16];  // [owned column][k]
};
struct FragAcc {
  float v[8];
};

__device__ __forceinline__ void barrier() { __syncthreads(); }
__device__ __forceinline__ void pad_barrier() { barrier(); }

__device__ __forceinline__ void clear(FragAcc& c) {
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) c.v[i] = 0.0f;
}
template <bool kFp16 = false>
__device__ __forceinline__ void load_a(FragA& a, const std::uint16_t* p, int ld) {
  const std::uint16_t* row = p + row_of_thread() * ld;
  HIP_BLOCK_UNROLL
  for (int k = 0; k < 16; ++k) a.v[k] = detail::f(row[k]);
}
template <bool kFp16 = false>
__device__ __forceinline__ void load_b(FragB& b, const std::uint16_t* p, int ld) {
  const int c0 = col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) {
    const std::uint16_t* col = p + (c0 + i) * ld;
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 16; ++k) b.v[i][k] = detail::f(col[k]);
  }
}
template <bool kFp16 = false>
__device__ __forceinline__ void load_b_rows(FragB& b, const std::uint16_t* p, int ld) {
  const int c0 = col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) {
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 16; ++k) b.v[i][k] = detail::f(p[k * ld + c0 + i]);
  }
}
__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) {
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) {
    float acc = c.v[i];
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 16; ++k) acc = detail::add(acc, detail::mul(a.v[k], b.v[i][k]));
    c.v[i] = acc;
  }
}
__device__ __forceinline__ void store_acc(float* p, const FragAcc& c, int ld) {
  float* row = p + row_of_thread() * ld + col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) row[i] = c.v[i];
}
__device__ __forceinline__ void load_acc(FragAcc& c, const float* p, int ld) {
  const float* row = p + row_of_thread() * ld + col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) c.v[i] = row[i];
}

}  // namespace hip_block::leaf

#endif
