// Focused check of the high-tier INT8 GEMM against a host replay, with partial M/N tiles and an optional LoRA delta.
// Usage: hip_int8_gemm_check [M N K] (K must be a multiple of 16; defaults exercise several tail shapes).
// Exit status: 0 all cases match bit for bit, 1 a mismatch, 2 no usable device.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "half.h"
#include "hip_block.h"
#include "int8_gemm.h"
#include "runtime.h"

namespace {

using bf16 = std::uint16_t;

bf16 rbf(float v) { return mi::hip::float_to_bf16(v); }

struct Case {
  int m, n, k;
};

bool check(const mi::hip::Device& device, const Case& c, bool delta) {
  std::mt19937 gen(12345 + c.m * 131 + c.n * 17 + c.k + (delta ? 1 : 0));
  std::uniform_int_distribution<int> q(-127, 127), sd(0, 255);
  const std::size_t xb = static_cast<std::size_t>(c.m) * c.k, wb = static_cast<std::size_t>(c.n) * c.k;
  std::vector<std::int8_t> xh(xb), wh(wb);
  for (auto& v : xh) v = static_cast<std::int8_t>(q(gen));
  for (auto& v : wh) v = static_cast<std::int8_t>(q(gen));
  std::vector<float> xsh(c.m), wsh(c.n);
  for (auto& v : xsh) v = 0.001f + static_cast<float>(sd(gen)) * 0.0001f;
  for (auto& v : wsh) v = 0.001f + static_cast<float>(sd(gen)) * 0.0001f;
  std::vector<bf16> dh(delta ? static_cast<std::size_t>(c.m) * c.n : 0);
  for (auto& v : dh) v = rbf(static_cast<float>(q(gen)) * 0.0001f);

  mi::hip::DeviceBuffer dx(xb), dw(wb), dy(static_cast<std::size_t>(c.m) * c.n * 2);
  mi::hip::DeviceBuffer dxs(c.m * 4), dws(c.n * 4), dd(dh.size() * 2);
  dx.upload(xh.data(), xb);
  dw.upload(wh.data(), wb);
  dxs.upload(xsh.data(), xsh.size() * 4);
  dws.upload(wsh.data(), wsh.size() * 4);
  if (delta) dd.upload(dh.data(), dh.size() * 2);

  mi::hip::int8_gemm(dy.as<bf16>(), c.n, dx.as<std::int8_t>(), c.k, dxs.as<float>(), dw.as<std::int8_t>(),
                     dws.as<float>(), nullptr, c.m, c.n, c.k, nullptr, delta ? dd.as<bf16>() : nullptr, delta ? c.n : 0);
  device.synchronize();
  std::vector<bf16> got(static_cast<std::size_t>(c.m) * c.n);
  dy.download(got.data(), got.size() * 2);

  std::size_t bad = 0;
  for (int r = 0; r < c.m; ++r) {
    for (int col = 0; col < c.n; ++col) {
      std::int32_t acc = 0;
      for (int t = 0; t < c.k; ++t) acc += static_cast<std::int32_t>(xh[static_cast<std::size_t>(r) * c.k + t]) *
                                          static_cast<std::int32_t>(wh[static_cast<std::size_t>(col) * c.k + t]);
      float v = static_cast<float>(acc) * xsh[r] * wsh[col];
      if (delta) v += mi::hip::bf16_to_float(dh[static_cast<std::size_t>(r) * c.n + col]);
      if (got[static_cast<std::size_t>(r) * c.n + col] != rbf(v)) ++bad;
    }
  }
  std::printf("  %4dx%4dx%4d delta=%d: %s (%zu mismatched of %d)\n", c.m, c.n, c.k, static_cast<int>(delta),
              bad == 0 ? "ok" : "FAILED", bad, c.m * c.n);
  return bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
  mi::hip::Device device(-1);
  std::vector<Case> cases;
  if (argc == 4) {
    cases.push_back({std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])});
  } else {
    cases = {{37, 80, 64}, {65, 96, 64}, {1, 32, 32}, {129, 160, 32}, {64, 32, 32}, {17, 17, 32},
             {65, 96, 4096}, {37, 80, 8192}, {256, 512, 3072}};
  }
  bool ok = true;
  for (const Case& c : cases) {
    if (c.k <= 0 || c.k % 16 != 0 || c.m <= 0 || c.n <= 0) {
      std::fprintf(stderr, "M, N and K must be positive and K a multiple of 16\n");
      return 2;
    }
    ok &= check(device, c, false);
    ok &= check(device, c, true);
  }
  return ok ? 0 : 1;
}
