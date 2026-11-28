// The INT8 leaf of the high tier's ConvRot GEMM: a 16x16x16 int8 WMMA with an int32 accumulator, per wave.
//
//   gfx11 (RDNA3)  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32, raw intrinsics
//   gfx10 (RDNA2)  __builtin_amdgcn_sdot4 (V_DOT4_I32_I8), one wave per 16x16x16 tile
//
// No rocWMMA: its fragment and accumulator layouts are opaque, and its load_matrix_sync/store_matrix_sync helpers
// force a particular LDS layout that prevents the multi-warp, software-pipelined tiled kernel. The raw gfx11 layout is
// public and simple: a 16-byte K-step fragment is held whole by every lane (row = lane % 16), and accumulator element e
// is row 2*e + lane/16, column lane%16. This matches the ROCm-optimised ComfyUI kernels on gfx1103.
//
// The tiled GEMM stages operands through LDS and writes its epilogue itself, so the only leaf operations it needs are
// `clear` and `mma`. Every call is collective: all 32 threads of the wave make it with the same (thread-independent)
// arguments and under the same control flow. The kernels depend on lockstep: they reuse a __shared__ buffer right after
// a collective read of it, with a barrier before the next write's readers.
#pragma once

#if defined(ROCWMMA_I8_LEAF) && !defined(__GFX10__)

#include <cstdint>

namespace hip_block::i8leaf {

using v4i = int __attribute__((ext_vector_type(4)));
using v8i = int __attribute__((ext_vector_type(8)));

using FragA = v4i;
using FragB = v4i;
using FragAcc = v8i;

__device__ __forceinline__ void clear(FragAcc& c) { c = {0, 0, 0, 0, 0, 0, 0, 0}; }

__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) {
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

}  // namespace hip_block::i8leaf

#elif defined(ROCWMMA_I8_LEAF) && defined(__GFX10__)

// RDNA2 has no WMMA, but it has V_DOT4_I32_I8. A lane's eight int32 accumulators are still element e = row
// 2*e + lane/16, column lane%16 of a 16x16 tile. Rather than holding one A row and shuffling seven more out of the
// other lanes, each lane loads the eight A rows it needs straight from LDS: the 16 lanes of a half-wave all read the
// same eight 16-byte rows, so those are LDS broadcasts, and the eight rows are 16-byte aligned and 20 words apart
// (kStride 80), i.e. bank-conflict free. Four dot4 per accumulator then consume the 16 int8 products of a K-step.
namespace hip_block::i8leaf {

using v4i = int __attribute__((ext_vector_type(4)));
using v8i = int __attribute__((ext_vector_type(8)));

struct FragA {
  v4i rows[8];  // the eight A rows this lane's accumulator elements need, one 16-byte K-step each
};
using FragB = v4i;
using FragAcc = v8i;

// row0 is the fragment's first A row; half is lane/16 (0: even rows, 1: odd rows).
__device__ __forceinline__ void load_a(FragA& a, const std::int8_t* lds, int row0, int kbyte, int stride, int half) {
#pragma unroll
  for (int e = 0; e < 8; ++e)
    a.rows[e] = *reinterpret_cast<const v4i*>(lds + (row0 + 2 * e + half) * stride + kbyte);
}

__device__ __forceinline__ void clear(FragAcc& c) { c = {0, 0, 0, 0, 0, 0, 0, 0}; }

__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) {
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int acc = c[e];
    acc = __builtin_amdgcn_sdot4(a.rows[e][0], b[0], acc, true);
    acc = __builtin_amdgcn_sdot4(a.rows[e][1], b[1], acc, true);
    acc = __builtin_amdgcn_sdot4(a.rows[e][2], b[2], acc, true);
    acc = __builtin_amdgcn_sdot4(a.rows[e][3], b[3], acc, true);
    c[e] = acc;
  }
}

}  // namespace hip_block::i8leaf

#endif
