// Tiled bf16 GEMM for the LoRA side branch and, by default, the text encoder's projections:
// C[m,n] = A[m,k] @ B[n,k]^T, fp32 accumulation with a single bf16 rounding at the store.
// The op order and rounding match hip_block::gemm, but a 128x128 LDS-staged tile replaces its 32x48 one.
#pragma once

#include <cstdint>

#include "hip_block.h"

namespace mi::hip {

struct Bf16GemmArgs {
  const hip_block::bf16_bits* a;
  const hip_block::bf16_bits* b;
  hip_block::bf16_bits* c;
  const hip_block::bf16_bits* residual;  // optional [m, ldc], added before the bf16 store
  std::int64_t lda, ldb, ldc;
  int m, n, k;
  int m_block0 = 0;  // first M block of this chunk when the grid is split over M
};

void bf16_gemm_tiled(hip_block::bf16_bits* c, std::int64_t ldc,
                     const hip_block::bf16_bits* a, std::int64_t lda,
                     const hip_block::bf16_bits* b, std::int64_t ldb,
                     int m, int n, int k, hipStream_t stream,
                     const hip_block::bf16_bits* residual = nullptr);

}  // namespace mi::hip
