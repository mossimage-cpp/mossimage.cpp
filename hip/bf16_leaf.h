// The VAE's raw bf16 WMMA leaf on gfx11/gfx12. One wave cooperates on a 16x16x16 bf16 multiply with an fp32
// accumulator; the layout is the intrinsic's public one (the same as bf16_gemm.hip): lane r holds one A row and one B
// row of 16 bf16 values, and accumulator element e is row 2*e + lane/16, column lane%16.
//
// Deliberately NOT rocWMMA: its fragment types and load_matrix_sync helpers force an LDS layout that is slower than
// the raw intrinsic here (the same finding that led int8_leaf.h and bf16_gemm.hip to use raw gfx11 WMMA). The VAE's
// kernels already stage A into LDS and store their own epilogues, so a leaf needs only load_a/load_b/mma/store_acc.
//
// Every call is collective: all 32 lanes of the wave make it under the same control flow. Callers launch one wave per
// workgroup, so a workgroup barrier is sufficient to order the shared-memory accesses around these calls.
#pragma once

#include <cstdint>

#if defined(HIP_BLOCK_LEAF_ROCWMMA) && !defined(__GFX10__)

#include <hip/hip_runtime.h>

namespace hip_vae::rleaf {

using v16bf = __bf16 __attribute__((ext_vector_type(16)));
using v8f = float __attribute__((ext_vector_type(8)));

using FragA = v16bf;
using FragB = v16bf;
using FragAcc = v8f;

inline constexpr const char* kName = "raw gfx11 WMMA bf16 (VAE)";

__device__ __forceinline__ int lane() { return static_cast<int>(threadIdx.x) & 31; }
__device__ __forceinline__ int frag_row() { return lane() & 15; }  // the 16 rows of a fragment map to lanes 0..15

__device__ __forceinline__ void barrier() { __syncthreads(); }
__device__ __forceinline__ void pad_barrier() { __syncthreads(); }

__device__ __forceinline__ void clear(FragAcc& c) { c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; }

// p points at a packed 16x16 fragment; ld is 16 elements. The row is selected by the lane. kFp16 is accepted for
// interface compatibility with mma_leaf.h; the operands are already bf16 bits here.
template <bool kFp16 = false>
__device__ __forceinline__ void load_a(FragA& a, const std::uint16_t* p, int ld) {
  a = *reinterpret_cast<const v16bf*>(p + static_cast<std::int64_t>(frag_row()) * ld);
}
// p points at a [n, k] weight fragment: one output channel's 16 bf16 K values per row.
template <bool kFp16 = false>
__device__ __forceinline__ void load_b(FragB& b, const std::uint16_t* p, int ld) {
  b = *reinterpret_cast<const v16bf*>(p + static_cast<std::int64_t>(frag_row()) * ld);
}

__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) {
  c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// store_acc/load_acc use the intrinsic's accumulator layout: element e at (row 2*e + lane/16, col lane%16).
__device__ __forceinline__ void store_acc(float* p, const FragAcc& c, int ld) {
  const int l = lane();
  float* q = p + static_cast<std::int64_t>(l >> 4) * ld + (l & 15);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    q[0] = c[e];
    q += 2 * ld;
  }
}
__device__ __forceinline__ void load_acc(FragAcc& c, const float* p, int ld) {
  const int l = lane();
  const float* q = p + static_cast<std::int64_t>(l >> 4) * ld + (l & 15);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    c[e] = q[0];
    q += 2 * ld;
  }
}

}  // namespace hip_vae::rleaf

#else

// The bf16 operand mode is only launched on gfx11/gfx12. Keep the name valid in the other code objects so the host
// launcher (which cannot see __GFX10__) still compiles the template instantiations.
#include "mma_leaf.h"
namespace hip_vae {
namespace rleaf = hip_block::leaf;
}  // namespace hip_vae

#endif  // gfx11/gfx12 raw bf16 WMMA
