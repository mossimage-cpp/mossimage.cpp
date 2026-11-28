// Compare the fast ConvRot dequantization with the reference on random rows.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "dequant_fast.h"

int main() {
  std::mt19937 rng(1);
  std::uniform_int_distribution<int> q(-127, 127);
  for (int g : {4, 16, 64, 256}) {
    const int rows = 32, k = 1024;
    std::vector<std::int8_t> quant(static_cast<std::size_t>(rows) * k);
    std::vector<float> scale(rows);
    for (auto& v : quant) v = static_cast<std::int8_t>(q(rng));
    for (auto& v : scale) v = 0.001f + 0.01f * std::uniform_real_distribution<float>(0.f, 1.f)(rng);
    std::vector<float> want(static_cast<std::size_t>(rows) * k), got(static_cast<std::size_t>(rows) * k);
    auto t0 = std::chrono::steady_clock::now();
    mi::quant::dequantize_rows(quant.data(), scale.data(), rows, k, g, want.data());
    double slow = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    mi::hip::dequantize_rows_fast(quant.data(), scale.data(), rows, k, g, got.data());
    double fast = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double worst = 0.0, sum = 0.0;
    std::size_t bad_bf16 = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
      worst = std::max(worst, d);
      sum += d;
      if (mi::quant::float_to_bf16(got[i]) != mi::quant::float_to_bf16(want[i])) ++bad_bf16;
    }
    std::printf("g=%d ref %.3fs fast %.4fs speedup %.1fx  worst=%.3g mean=%.3g bf16diff=%zu/%zu\n", g, slow, fast,
                slow / fast, worst, sum / got.size(), bad_bf16, got.size());
  }
  return 0;
}
