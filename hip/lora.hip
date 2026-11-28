// LoRA support has no dedicated device kernels: the low-rank factors are ordinary bf16 GEMMs through hip_block::gemm,
// and the delta is folded into the base INT8 GEMM's fp32 accumulator before its bf16 rounding (see int8_gemm.hip).
// This translation unit is kept so the kernel source list stays uniform across backends.
#include <hip/hip_runtime.h>
