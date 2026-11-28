// sm_75 INT8 ConvRot GEMMs (see kernels/sm75/int8_sm75.h): native tiled __dp4a GEMMs with bit-exact int32
// accumulation and the same dequantisation order as the sm_80 CUTLASS path.
#include "kernels/sm75/int8_sm75.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "core/tensor.h"
#include "kernels/basic/elementwise.h"
#include "kernels/quant/int8_gemm_impl.cuh"

namespace mi::kernels::sm75 {
namespace qd = quant::detail;

namespace {

// Native DP4A tiled GEMM for the plain epilogue. Turing's m8n8k16 INT8 MMA is decoded on ordinary CUDA cores here
// (GTX 16xx has no tensor cores); four int8 MACs per dp4a instruction with a 64x32x128 tile and shared weight reuse
// is faster than the emulated MMA. W stays in the engine's [N,K] layout: global loads run along K (coalesced) and
// store into a transposed [K,N] shared tile. The int32 dot product and dequantisation order are unchanged.
template <int BM, int BN, int BK, bool kAddDelta>
__global__ void dp4a_plain_kernel(quant::Int8Operands op, const float* __restrict__ xs,
                                  const float* __restrict__ ws, const __nv_bfloat16* __restrict__ bias,
                                  __nv_bfloat16* __restrict__ out, std::int64_t ldo,
                                   const __nv_bfloat16* __restrict__ delta, std::int64_t ldd) {
  __shared__ std::int8_t As[BM][BK];
  __shared__ std::int8_t Bs[BK][BN];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
  int acc[BM / 8];
#pragma unroll
  for (int i = 0; i < BM / 8; ++i) acc[i] = 0;
  for (int k0 = 0; k0 < op.k; k0 += BK) {
    for (int i = ty; i < BM; i += 8) {
      const std::int64_t r = r0 + i;
      if (r >= op.m) continue;
      for (int j = tx; j < BK; j += 32) {
        const int k = k0 + j;
        As[i][j] = k < op.k ? op.x_q[r * op.ldx + k] : 0;
      }
    }
    for (int i = ty; i < BN; i += 8) {
      const int c = c0 + i;
      if (c >= op.n) continue;
      for (int j = tx; j < BK; j += 32) {
        const int k = k0 + j;
        Bs[j][i] = k < op.k ? op.w[static_cast<std::size_t>(c) * op.k + k] : 0;
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k += 4) {
      int a[BM / 8];
#pragma unroll
      for (int i = 0; i < BM / 8; ++i) a[i] = *reinterpret_cast<const int*>(&As[ty + i * 8][k]);
      const int b = static_cast<unsigned char>(Bs[k][tx]) | (static_cast<unsigned char>(Bs[k + 1][tx]) << 8) |
                    (static_cast<unsigned char>(Bs[k + 2][tx]) << 16) |
                    (static_cast<unsigned char>(Bs[k + 3][tx]) << 24);
#pragma unroll
      for (int i = 0; i < BM / 8; ++i) acc[i] = __dp4a(a[i], b, acc[i]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < BM / 8; ++i) {
    const std::int64_t r = r0 + ty + i * 8, c = c0 + tx;
    if (r >= op.m || c >= op.n) continue;
    const float b = bias != nullptr ? static_cast<float>(bias[c]) : 0.f;
    const float v = qd::dequant(__int2float_rn(acc[i]), xs[r], ws[c], b);
    __nv_bfloat16 bv = __float2bfloat16_rn(v);
    if constexpr (kAddDelta) {
      // Round the base to bf16, add the bf16 delta in fp32, round once — the same order as a separate add-back pass.
      const float sum = __bfloat162float(bv) + __bfloat162float(delta[r * ldd + c]);
      bv = __float2bfloat16_rn(sum);
    }
    out[r * ldo + c] = bv;
  }
}

void plain(const qd::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* bias, __nv_bfloat16* out,
           std::int64_t ldo, cudaStream_t stream) {
  constexpr int BM = 64, BN = 32, BK = 128;
  const dim3 grid((p.n + BN - 1) / BN, (p.m + BM - 1) / BM);
  dp4a_plain_kernel<BM, BN, BK, false><<<grid, dim3(32, 8), 0, stream>>>(a, a.x_scale, a.w_scale, bias, out, ldo,
                                                                        nullptr, 0);
  MI_CUDA_CHECK_LAUNCH();
}

// Fused epilogues: run the GEMM with native DP4A into a temporary bf16 matrix, then apply the bit-exact SIMT
// epilogue. One scratch buffer per stream: a stream's calls are ordered, and separate streams never share a buffer.
// sm_75 does not capture these blocks into CUDA graphs.
void* scratch(cudaStream_t stream, std::size_t bytes) {
  static std::mutex mutex;
  static std::unordered_map<cudaStream_t, std::pair<void*, std::size_t>> buffers;
  std::lock_guard<std::mutex> lock(mutex);
  auto& b = buffers[stream];
  if (b.second < bytes) {
    if (b.first != nullptr) MI_CUDA_CHECK(cudaFree(b.first));
    b.first = nullptr;
    MI_CUDA_CHECK(cudaMalloc(&b.first, bytes));
    b.second = bytes;
  }
  return b.first;
}

Tensor bf16_view(void* p, std::int64_t rows, std::int64_t cols, std::int64_t ld) {
  Tensor t;
  t.data = p;
  t.dtype = DType::kBF16;
  t.device = Device::kCUDA;
  t.rank = 2;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = ld;
  t.stride[1] = 1;
  return t;
}

// Fused LoRA epilogue: the base DP4A GEMM adds the already-computed bf16 delta in its own epilogue, replacing a
// separate elementwise pass with identical bf16 rounding order.
void plain_add(const qd::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* delta, std::int64_t ldd,
               __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  constexpr int BM = 64, BN = 32, BK = 128;
  const dim3 grid((p.n + BN - 1) / BN, (p.m + BM - 1) / BM);
  dp4a_plain_kernel<BM, BN, BK, true><<<grid, dim3(32, 8), 0, stream>>>(a, a.x_scale, a.w_scale, nullptr, out, ldo,
                                                                       delta, ldd);
  MI_CUDA_CHECK_LAUNCH();
}

void gated_residual(const qd::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* bias,
                    const __nv_bfloat16* gate, __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  if (p.m == 0) return;
  auto* tmp = static_cast<__nv_bfloat16*>(scratch(stream, static_cast<std::size_t>(p.m) * p.n * 2));
  plain(p, a, bias, tmp, p.n, stream);
  const Tensor t = bf16_view(tmp, p.m, p.n, p.n);
  const Tensor g = bf16_view(const_cast<__nv_bfloat16*>(gate), 1, p.n, p.n);
  const Tensor r = bf16_view(residual, p.m, p.n, ldr);
  kernels::gated_residual(r, t, g, r, stream);
}

void situ_glu(const qd::Problem& p, const quant::Int8Operands& a, __nv_bfloat16* out, std::int64_t ldo,
              cudaStream_t stream) {
  if (p.m == 0) return;
  auto* tmp = static_cast<__nv_bfloat16*>(scratch(stream, static_cast<std::size_t>(p.m) * p.n * 2));
  plain(p, a, nullptr, tmp, p.n, stream);
  const Tensor t = bf16_view(tmp, p.m, p.n, p.n);
  const Tensor o = bf16_view(out, p.m, p.n / 2, ldo);
  kernels::situ_glu(t, o, stream);
}

}  // namespace

void int8_gemm(const quant::Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "out", out, ldo, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  if (p.m == 0) return;
  plain(p, op, bias, out, ldo, stream);
}

void int8_gemm_add_delta(const quant::Int8Operands& op, const __nv_bfloat16* delta, std::int64_t ldd,
                         __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm_add_delta";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "out", out, ldo, p.n);
  qd::check_vector(name, "delta", delta, /*optional=*/false);
  if (p.m == 0) return;
  plain_add(p, op, delta, ldd, out, ldo, stream);
}

void int8_gemm_gated_residual(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm_gated_residual";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "residual", residual, ldr, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  qd::check_vector(name, "gate", gate, /*optional=*/false);
  if (p.m == 0) return;
  gated_residual(p, op, bias, gate, residual, ldr, stream);
}

void int8_gemm_situ_glu(const quant::Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm_situ_glu";
  if (op.n % (2 * qd::kUpGateGroup) != 0) {
    kernels::detail::fail(name, "n = 2F needs F % 4 == 0, got n " + std::to_string(op.n));
  }
  const qd::Problem p = qd::check_operands(name, op);
  if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 8 != 0 || ldo < p.n / 2 || ldo % 4 != 0) {
    kernels::detail::fail(name, "out must be 8-byte aligned with a row stride >= F and a multiple of 4, got " +
                                    std::to_string(ldo));
  }
  if (p.m == 0) return;
  situ_glu(p, op, out, ldo, stream);
}

}  // namespace mi::kernels::sm75
