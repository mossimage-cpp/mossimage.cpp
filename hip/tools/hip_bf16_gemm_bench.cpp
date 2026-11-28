// Time the LoRA side-branch bf16 GEMM shapes.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "bf16_gemm.h"
#include "half.h"
#include "runtime.h"

using namespace mi::hip;
using bf16 = hip_block::bf16_bits;

int main() {
  Device device(-1);
  struct Shape {
    int m, n, k;
    const char* name;
  };
  const Shape shapes[] = {{6208, 384, 3072, "qkv.t"},   {6208, 9216, 384, "qkv.delta"},
                          {6208, 128, 3072, "o.t"},    {6208, 3072, 128, "o.delta"},
                          {6208, 128, 3072, "up.t"},   {6208, 16384, 128, "up.delta"},
                          {6208, 128, 8192, "down.t"}, {6208, 3072, 128, "down.delta"}};
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 0.01f);
  for (const Shape& s : shapes) {
    std::vector<bf16> ah(static_cast<std::size_t>(s.m) * s.k), bh(static_cast<std::size_t>(s.n) * s.k);
    for (auto& v : ah) v = float_to_bf16(nd(rng));
    for (auto& v : bh) v = float_to_bf16(nd(rng));
    DeviceBuffer da(ah.size() * 2), db(bh.size() * 2), dc(static_cast<std::size_t>(s.m) * s.n * 2);
    da.upload(ah.data(), ah.size() * 2);
    db.upload(bh.data(), bh.size() * 2);
    bf16_gemm_tiled(dc.as<bf16>(), s.n, da.as<bf16>(), s.k, db.as<bf16>(), s.k, s.m, s.n, s.k, 0);
    hipDeviceSynchronize();
    const int iters = 6;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      bf16_gemm_tiled(dc.as<bf16>(), s.n, da.as<bf16>(), s.k, db.as<bf16>(), s.k, s.m, s.n, s.k, 0);
    hipDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%-11s %5dx%-5dx%-5d %7.2f ms  %6.1f GMAC/s\n", s.name, s.m, s.n, s.k, ms / iters,
                2e-9 * s.m * s.n * s.k / (ms / iters * 1e-3));
  }
  return 0;
}
