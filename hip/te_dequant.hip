#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "te_dequant.h"

namespace mi::hip::te_dequant {
namespace {

using namespace hip_block::detail;

constexpr int kWave = 32;
constexpr int kGroup = 256;  // ConvRot's Hadamard group

struct DequantArgs {
  const std::int8_t* q;
  const float* scale;
  std::uint16_t* out;
  int k, groups;
};

// Four waves per workgroup, one 256-element group per wave. The four stages read a disjoint butterfly's four values
// before writing them, so no barriers are needed inside a stage; the group's scale multiply is the same order as
// dequant_fast.h's host code. Four independent groups in one workgroup cut the block count (and launch overhead) 4x.
constexpr int kGroupsPerBlock = 4;

__global__ __launch_bounds__(kWave * kGroupsPerBlock) void dequant_i8_kernel(const DequantArgs a) {
  __shared__ float v[kGroupsPerBlock][kGroup];
  const int wave = static_cast<int>(threadIdx.x) / kWave;
  const int t = static_cast<int>(threadIdx.x) % kWave;
  const int groups_per_row = a.k / kGroup;
  for (int base = static_cast<int>(blockIdx.x) * kGroupsPerBlock; base < a.groups;
       base += static_cast<int>(gridDim.x) * kGroupsPerBlock) {
    const int g = base + wave;
    const bool valid = g < a.groups;
    const int gg = valid ? g : 0;  // idle waves still take every barrier, with a harmless duplicate of group 0
    const float sc = valid ? a.scale[gg / groups_per_row] : 0.0f;
    const std::int64_t at = static_cast<std::int64_t>(gg) * kGroup;
    HIP_BLOCK_UNROLL
    for (int i = 0; i < kGroup / kWave; ++i) {
      const int j = t * (kGroup / kWave) + i;
      v[wave][j] = static_cast<float>(a.q[at + j]) * sc;
    }
    __syncthreads();
    for (int step = 1; step < kGroup; step <<= 2) {
      // 64 butterflies per stage, two per lane.
      HIP_BLOCK_UNROLL
      for (int b = t * (kGroup / (4 * kWave)); b < t * (kGroup / (4 * kWave)) + kGroup / (4 * kWave); ++b) {
        const int ibase = (b / step) * 4 * step, off = b % step;
        const float w0 = v[wave][ibase + off], w1 = v[wave][ibase + step + off];
        const float w2 = v[wave][ibase + 2 * step + off], w3 = v[wave][ibase + 3 * step + off];
        const float p = w0 - w3, q = w0 + w3, r = w1 + w2, s = w1 - w2;
        v[wave][ibase + off] = 0.5f * (p + r);
        v[wave][ibase + step + off] = 0.5f * (q + s);
        v[wave][ibase + 2 * step + off] = 0.5f * (q - s);
        v[wave][ibase + 3 * step + off] = 0.5f * (r - p);
      }
      __syncthreads();
    }
    if (valid) {
      HIP_BLOCK_UNROLL
      for (int i = 0; i < kGroup / kWave; ++i) {
        const int j = t * (kGroup / kWave) + i;
        a.out[at + j] = bf(v[wave][j]);
      }
    }
    __syncthreads();
  }
}

}  // namespace

void dequant_i8(const std::int8_t* q, const float* scale, std::uint16_t* out, int n, int k, hipStream_t stream) {
  if (n <= 0 || k <= 0) return;
  const int groups = n * (k / kGroup);
  const DequantArgs a{q, scale, out, k, groups};
  const int blocks = (groups + kGroupsPerBlock - 1) / kGroupsPerBlock;
  dequant_i8_kernel<<<dim3(blocks < 4096 ? blocks : 4096), dim3(kWave * kGroupsPerBlock), 0, stream>>>(a);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("te_dequant: ") + hipGetErrorString(e));
}

}  // namespace mi::hip::te_dequant
