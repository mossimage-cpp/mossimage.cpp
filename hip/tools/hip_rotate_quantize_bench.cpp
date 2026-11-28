// Time the activation quantization that precedes each INT8 GEMM.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "runtime.h"

namespace mi::hip {
void rotate_quantize(const std::uint16_t* x, std::int64_t ldx, std::int64_t rows, std::int64_t k,
                     std::int8_t* q, std::int64_t ldq, float* scale, hipStream_t stream);
}

int main() {
  mi::hip::Device device(-1);
  const std::int64_t rows = 6208;
  for (int k : {3072, 8192}) {
    const std::size_t n = static_cast<std::size_t>(rows) * k;
    mi::hip::DeviceBuffer dx(n * 2), dq(n), ds(rows * 4);
    dx.upload(std::vector<std::uint16_t>(n, 0x3f00).data(), n * 2);
    const hipStream_t st = 0;
    for (int w = 0; w < 2; ++w) mi::hip::rotate_quantize(dx.as<std::uint16_t>(), k, rows, k,
                                                         dq.as<std::int8_t>(), k, ds.as<float>(), st);
    hipDeviceSynchronize();
    const int iters = 8;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      mi::hip::rotate_quantize(dx.as<std::uint16_t>(), k, rows, k, dq.as<std::int8_t>(), k, ds.as<float>(), st);
    hipDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("rotate_quantize rows=%lld k=%d: %.3f ms\n", static_cast<long long>(rows), k, ms / iters);
  }
  return 0;
}
