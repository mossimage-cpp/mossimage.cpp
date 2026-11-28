// Check the tiled bf16 GEMM against a host replay on LoRA-like and tail shapes.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bf16_gemm.h"
#include "half.h"
#include "runtime.h"

using namespace mi::hip;
using bf16 = hip_block::bf16_bits;

static bf16 rbf(float v) { return mi::hip::float_to_bf16(v); }

int main() {
  Device device(-1);
  std::mt19937 gen(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int cases[][3] = {{37, 80, 64}, {65, 96, 128}, {129, 160, 128}, {256, 512, 128}, {521, 128, 3072}};
  bool ok = true;
  for (const auto& c : cases) {
    const int m = c[0], n = c[1], k = c[2];
    std::vector<bf16> ah(static_cast<std::size_t>(m) * k), bh(static_cast<std::size_t>(n) * k);
    for (auto& v : ah) v = rbf(nd(gen));
    for (auto& v : bh) v = rbf(nd(gen));
    DeviceBuffer da(ah.size() * 2), db(bh.size() * 2), dc(static_cast<std::size_t>(m) * n * 2);
    da.upload(ah.data(), ah.size() * 2);
    db.upload(bh.data(), bh.size() * 2);
    bf16_gemm_tiled(dc.as<bf16>(), n, da.as<bf16>(), k, db.as<bf16>(), k, m, n, k, 0);
    device.synchronize();
    std::vector<bf16> got(static_cast<std::size_t>(m) * n);
    dc.download(got.data(), got.size() * 2);
    // Different K-step and fragment orders are valid as long as the result is within bf16 rounding distance of the
    // fp32 reference.
    std::size_t bad = 0;
    for (int r = 0; r < m; ++r) {
      for (int col = 0; col < n; ++col) {
        float acc = 0.0f;
        for (int t = 0; t < k; ++t) {
          const float av = mi::hip::bf16_to_float(ah[static_cast<std::size_t>(r) * k + t]);
          const float bv = mi::hip::bf16_to_float(bh[static_cast<std::size_t>(col) * k + t]);
          acc += av * bv;
        }
        const float got_v = mi::hip::bf16_to_float(got[static_cast<std::size_t>(r) * n + col]);
        if (std::fabs(got_v - acc) > 0.01f * (1.0f + std::fabs(acc))) ++bad;
      }
    }
    std::printf("%4dx%4dx%4d: %s (%zu of %d)\n", m, n, k, bad ? "FAILED" : "ok", bad, m * n);
    ok &= bad == 0;
  }
  return ok ? 0 : 1;
}
