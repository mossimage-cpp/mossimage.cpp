// Time the high tier's four INT8 GEMM shapes with the build's tile configuration.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "int8_gemm.h"
#include "runtime.h"

using namespace mi::hip;
using bf16 = hip_block::bf16_bits;

int main() {
  Device device(-1);
  struct Shape {
    int m, n, k;
  };
  const Shape shapes[] = {{6208, 9216, 3072}, {6208, 3072, 3072}, {6208, 16384, 3072}, {6208, 3072, 8192}};
  for (const Shape& s : shapes) {
    const std::size_t nb = static_cast<std::size_t>(s.n) * s.k;
    const std::size_t xb = static_cast<std::size_t>(s.m) * s.k;
    const std::size_t yb = static_cast<std::size_t>(s.m) * s.n;
    DeviceBuffer dx(xb), dw(nb), dy(yb * 2), ds(s.m * sizeof(float)), dws(s.n * sizeof(float));
    dx.upload(std::vector<std::int8_t>(xb, 3).data(), xb);
    dw.upload(std::vector<std::int8_t>(nb, 3).data(), nb);
    ds.upload(std::vector<float>(s.m, 0.02f).data(), s.m * sizeof(float));
    dws.upload(std::vector<float>(s.n, 0.02f).data(), s.n * sizeof(float));
    const hipStream_t st = 0;
    for (int w = 0; w < 2; ++w) int8_gemm(dy.as<bf16>(), s.n, dx.as<std::int8_t>(), s.k, ds.as<float>(),
                                          dw.as<std::int8_t>(), dws.as<float>(), nullptr, s.m, s.n, s.k, st);
    hipDeviceSynchronize();
    const int iters = 8;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      int8_gemm(dy.as<bf16>(), s.n, dx.as<std::int8_t>(), s.k, ds.as<float>(), dw.as<std::int8_t>(),
                dws.as<float>(), nullptr, s.m, s.n, s.k, st);
    hipDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%5dx%-5dx%-5d %.2f ms\n", s.m, s.n, s.k, ms / iters);
  }
  return 0;
}
