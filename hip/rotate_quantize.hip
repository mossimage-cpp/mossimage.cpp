// The high tier's activation quantization: rotate a row's 256-groups by a Hadamard and quantize the row to int8 with one
// per-row scale, as the engine's kernels/quant/rotate_quantize.cu. A half-warp holds one 256-group, 16 elements per lane,
// so a warp holds a pair of groups (2p, 2p + 1) at a time; warp w takes pairs w, w + W, ... (kPairsPerWarp of them). With
// e = d0 + 4 d1 + 16 d2 + 64 d3 the base-4 digits of an element's index in its group:
// - load layout: lane h (of the half) holds e = 16 h + j, j < 16;
// - the 16 x 16 register stages (d0, d1), a shared-tile transpose, then the two register stages (d2, d3);
// - every stage output is 0.5 * (((m0' + m1') + m2') + m3') with operand 3 negated; the bits do not depend on the layout.
//
// The row absmax is a max over every rotated element (exact in any order); the scale is max(min(amax, bf16max) / 127,
// 1e-30); the quantized bytes go back through the same transpose, so each lane stores its 16 contiguous bytes of a group
// with one 16-byte store.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"

namespace mi::hip {
namespace {

using namespace hip_block;
using bf16 = std::uint16_t;
using detail::add;
using detail::bf;
using detail::div;
using detail::f;
using detail::mul;
using detail::rbf;
using detail::rsqrt;

constexpr int kMaxWarps = 16;
constexpr int kConvRotGroup = 256;
constexpr int kStride = 17;               // shared-memory row stride of the transpose
constexpr int kHalfFloats = 16 * kStride;  // one half-warp's 16 x 16 tile (272 floats)
constexpr float kInv127 = 1.0f / 127.0f;
constexpr float kMinScale = 1e-30f;
constexpr float kBf16Max = 3.38953139e38f;

// One radix-4 butterfly on operands ordered by digit: y_d = 0.5 * (((m0' + m1') + m2') + m3') with operand 3 negated, in
// that order. add/mul are the exact fp32 round-once ops; a negated operand's FADD is a FSUB.
__device__ __forceinline__ void h4(float& a, float& b, float& c, float& d) {
  const float s = add(a, b);
  const float t = add(a, -b);
  const float y0 = mul(0.5f, add(add(s, c), -d));
  const float y1 = mul(0.5f, add(add(s, -c), d));
  const float y2 = mul(0.5f, add(add(t, c), d));
  const float y3 = mul(0.5f, add(add(-t, c), d));
  a = y0;
  b = y1;
  c = y2;
  d = y3;
}

// Two radix-4 stages over register digits: first j & 3 (stride 1 in j), then j >> 2 (stride 4 in j).
__device__ __forceinline__ void two_stages(float (&v)[16]) {
  HIP_BLOCK_UNROLL
  for (int q = 0; q < 4; ++q) h4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
  HIP_BLOCK_UNROLL
  for (int r = 0; r < 4; ++r) h4(v[r], v[r + 4], v[r + 8], v[r + 12]);
}

// x_rot = x * H256 for the half-warp's group, fp32: in = load layout (e = 16 h + j), out = e = h + 16 j. tile: this half's
// kHalfFloats shared floats. The whole warp calls it (warp-synchronous transpose).
__device__ __forceinline__ void hadamard256(float (&v)[16], float* tile, int h) {
  two_stages(v);
  HIP_BLOCK_UNROLL
  for (int j = 0; j < 16; ++j) tile[h * kStride + j] = v[j];
  __syncwarp();
  HIP_BLOCK_UNROLL
  for (int j = 0; j < 16; ++j) v[j] = tile[j * kStride + h];
  __syncwarp();  // the tile is reused by the next pair of this warp
  two_stages(v);
}

// Pairs of groups per warp and warps per block for a row of `groups` groups: at most 8 warps while 4 pairs per warp
// suffice, else up to kMaxWarps.
struct BlockShape {
  int warps;
  int pairs;
  static BlockShape of(int groups) {
    BlockShape bs;
    // One warp handles one pair of 256-groups per iteration. Use enough warps to cover every group in one iteration
    // (up to kMaxWarps) and only then more iterations; the old formula asked for nine pairs at k=8192, which the
    // launcher's template switch did not support.
    bs.warps = (groups + 1) / 2;
    if (bs.warps > kMaxWarps) bs.warps = kMaxWarps;
    bs.pairs = (groups + 2 * bs.warps - 1) / (2 * bs.warps);
    if (bs.pairs < 1) bs.pairs = 1;
    return bs;
  }
};

struct RotateArgs {
  const bf16* x;
  std::int64_t ldx;
  std::int64_t k;
  std::int8_t* q;
  std::int64_t ldq;
  float* scale;
  const bf16* gate;  // optional second half for the fused SiTU-GLU read (up_gate row: [hidden, gate])
  int split;         // gate offset = split
};

template <int kPairsPerWarp>
__global__ void rotate_quantize_kernel(RotateArgs a, int row0) {
  extern __shared__ float tiles[];  // warps x 2 x kHalfFloats
  __shared__ float warp_max[kMaxWarps];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;
  const int half = lane >> 4, h = lane & 15;
  const int row = row0 + static_cast<int>(blockIdx.y);
  const int groups = static_cast<int>(a.k) / kConvRotGroup;

  float v[kPairsPerWarp][16];
  float* tile = tiles + (warp * 2 + half) * kHalfFloats;
  const bf16* base = a.x + static_cast<std::int64_t>(row) * a.ldx;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < kPairsPerWarp; ++i) {
    const int g = 2 * (warp + i * warps) + half;
    if (g >= groups) continue;
    const bf16* src = base + static_cast<std::int64_t>(g) * kConvRotGroup + 16 * h;
    // Two 16-byte loads instead of sixteen 2-byte ones: this scaler is memory-bound on RDNA2.
    const uint4 lo = *reinterpret_cast<const uint4*>(src);
    const uint4 hi = *reinterpret_cast<const uint4*>(src + 8);
    const std::uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    if (a.gate != nullptr) {
      // Fused SiTU-GLU: read the gate half of the same row, round the product to bf16 exactly as situ_glu does, then
      // rotate and quantize it. Saves a full [rows, k] bf16 round trip per block.
      const bf16* gsrc = a.gate + static_cast<std::int64_t>(row) * a.ldx + a.split +
                         static_cast<std::int64_t>(g) * kConvRotGroup + 16 * h;
      const uint4 glo = *reinterpret_cast<const uint4*>(gsrc);
      const uint4 ghi = *reinterpret_cast<const uint4*>(gsrc + 8);
      const std::uint32_t gw[8] = {glo.x, glo.y, glo.z, glo.w, ghi.x, ghi.y, ghi.z, ghi.w};
      HIP_BLOCK_UNROLL
      for (int j = 0; j < 16; ++j) {
        const float hv = f(static_cast<bf16>(w[j >> 1] >> (16 * (j & 1))));
        const float gv = f(static_cast<bf16>(gw[j >> 1] >> (16 * (j & 1))));
        v[i][j] = detail::rbf(detail::situ_value(hv, gv));  // situ_glu rounds the product to bf16 first
      }
    } else {
      HIP_BLOCK_UNROLL
      for (int j = 0; j < 16; ++j) v[i][j] = f(static_cast<bf16>(w[j >> 1] >> (16 * (j & 1))));
    }
  }

  float amax = 0.0f;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < kPairsPerWarp; ++i) {
    if (2 * (warp + i * warps) >= groups) continue;  // warp-uniform: the whole warp transposes together
    hadamard256(v[i], tile, h);
    const int g = 2 * (warp + i * warps) + half;
    if (g >= groups) continue;
    HIP_BLOCK_UNROLL
    for (int j = 0; j < 16; ++j) {
      v[i][j] = rbf(v[i][j]);  // the rotated activation is a bf16 tensor
      amax = fmaxf(amax, fabsf(v[i][j]));
    }
  }
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffull, amax, off));
  if (lane == 0) warp_max[warp] = amax;
  __syncthreads();
  amax = warp_max[0];
  for (int w = 1; w < warps; ++w) amax = fmaxf(amax, warp_max[w]);

  // fmaxf drops NaN operands, so a NaN element never reaches amax and an inf one is clamped to the bf16 maximum.
  const float s = fmaxf(mul(fminf(amax, kBf16Max), kInv127), kMinScale);
  if (threadIdx.x == 0) a.scale[row] = s;
  const float s_bf = rbf(s);  // x / scale runs on bf16 tensors
  std::int8_t* qr = a.q + static_cast<std::int64_t>(row) * a.ldq;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < kPairsPerWarp; ++i) {
    if (2 * (warp + i * warps) >= groups) continue;
    const int g = 2 * (warp + i * warps) + half;
    HIP_BLOCK_UNROLL
    for (int j = 0; j < 16; ++j) {
      // torch.round: half to even. fmaxf maps NaN to -128; +-inf saturate.
      const float qv = nearbyintf(rbf(div(v[i][j], s_bf)));
      tile[j * kStride + h] = fminf(127.0f, fmaxf(-128.0f, qv));
    }
    __syncwarp();
    std::int8_t bytes[16];
    HIP_BLOCK_UNROLL
    for (int j = 0; j < 16; ++j) bytes[j] = static_cast<std::int8_t>(tile[h * kStride + j]);
    __syncwarp();  // the tile is reused by the next pair
    if (g < groups) *reinterpret_cast<uint4*>(qr + static_cast<std::int64_t>(g) * kConvRotGroup + 16 * h) = *reinterpret_cast<uint4*>(bytes);
  }
}

void check_launch(const char* what) {
  const auto e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

}  // namespace

// Rotate a [rows, k] bf16 row and quantize to [rows, k] int8 with one per-row scale.
void rotate_quantize(const bf16* x, std::int64_t ldx, std::int64_t rows, std::int64_t k, std::int8_t* q,
                     std::int64_t ldq, float* scale, hipStream_t stream) {
  if (k <= 0 || k % kConvRotGroup != 0 || rows < 0 || ldx < k || ldq < k || x == nullptr || q == nullptr ||
      scale == nullptr)
    throw std::invalid_argument("rotate_quantize: k must be a positive multiple of 256 and the strides at least k");
  const int groups = static_cast<int>(k) / kConvRotGroup;
  const BlockShape bs = BlockShape::of(groups);
  const int threads = 32 * bs.warps;
  const size_t smem = static_cast<size_t>(bs.warps) * 2 * kHalfFloats * sizeof(float);
  const RotateArgs args{x, ldx, k, q, ldq, scale, nullptr, 0};
  // Grid y is the token index; chunk it so a 4.2 MP image does not exceed the device's grid limit.
  constexpr std::int64_t kChunkRows = 8192;
  for (std::int64_t row0 = 0; row0 < rows; row0 += kChunkRows) {
    const int chunk = static_cast<int>(std::min<std::int64_t>(kChunkRows, rows - row0));
    const dim3 grid(1, static_cast<unsigned>(chunk), 1);
    switch (bs.pairs) {
      case 1: rotate_quantize_kernel<1><<<grid, dim3(threads), smem, stream>>>(args, static_cast<int>(row0)); break;
      case 2: rotate_quantize_kernel<2><<<grid, dim3(threads), smem, stream>>>(args, static_cast<int>(row0)); break;
      case 3: rotate_quantize_kernel<3><<<grid, dim3(threads), smem, stream>>>(args, static_cast<int>(row0)); break;
      case 4: rotate_quantize_kernel<4><<<grid, dim3(threads), smem, stream>>>(args, static_cast<int>(row0)); break;
      default: throw std::runtime_error("rotate_quantize: unsupported pairs");
    }
  }
  check_launch("rotate_quantize");
}

// Rotate-and-quantize bf16(situ_glu([rows, 2*k])) into int8 [rows, k]. The row stride of the source is 2*k; `k` is the
// length of each half. Numerically identical to situ_glu followed by rotate_quantize, without the intermediate tensor.
void rotate_quantize_situ(const bf16* up_gate, std::int64_t ld, std::int64_t rows, std::int64_t k, std::int8_t* q,
                          std::int64_t ldq, float* scale, hipStream_t stream) {
  if (k <= 0 || k % kConvRotGroup != 0 || rows < 0 || ld < 2 * k || ldq < k || up_gate == nullptr || q == nullptr ||
      scale == nullptr)
    throw std::invalid_argument("rotate_quantize_situ: k must be a positive multiple of 256 and the strides at least k");
  const int groups = static_cast<int>(k) / kConvRotGroup;
  const BlockShape bs = BlockShape::of(groups);
  const int threads = 32 * bs.warps;
  const size_t smem = static_cast<size_t>(bs.warps) * 2 * kHalfFloats * sizeof(float);
  const RotateArgs args{up_gate, ld, k, q, ldq, scale, up_gate, static_cast<int>(k)};
  constexpr std::int64_t kChunkRows = 8192;
  for (std::int64_t row0 = 0; row0 < rows; row0 += kChunkRows) {
    const int chunk = static_cast<int>(std::min<std::int64_t>(kChunkRows, rows - row0));
    const dim3 grid(1, static_cast<unsigned>(chunk), 1);
    switch (bs.pairs) {
      case 1: rotate_quantize_kernel<1><<<grid, dim3(threads), smem, stream>>>(args, static_cast<int>(row0)); break;
      case 2: rotate_quantize_kernel<2><<<grid, dim3(threads), smem, stream>>>(args, static_cast<int>(row0)); break;
      case 3: rotate_quantize_kernel<3><<<grid, dim3(threads), smem, stream>>>(args, static_cast<int>(row0)); break;
      case 4: rotate_quantize_kernel<4><<<grid, dim3(threads), smem, stream>>>(args, static_cast<int>(row0)); break;
      default: throw std::runtime_error("rotate_quantize_situ: unsupported pairs");
    }
  }
  check_launch("rotate_quantize_situ");
}

}  // namespace mi::hip
