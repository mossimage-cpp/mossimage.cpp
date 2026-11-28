// Fast host dequantization of ConvRot int8 weights.
//
// mi::quant::dequantize_rows multiplies each 256-element group by a dense
// Hadamard table: O(g^2) scalar work per group, which dominates the text
// encoder's construction time. H_g = kron^n(H4) with H4[a][b] = -1 exactly when
// a + b == 3, so the product can be applied as n radix-4 butterfly stages in
// O(g log g). Each output is then rounded to bf16, the same as the caller did.
#pragma once

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include "weights/quant/quant_codec.h"

namespace mi::hip {
namespace dequant_detail {

inline void hadamard_group(float* v, int g) {
  // Base-4 digit d lives at stride 4^d. Apply the H4 butterfly to each 4-tuple.
  for (int step = 1; step < g; step <<= 2) {
    for (int base = 0; base < g; base += 4 * step) {
      for (int off = 0; off < step; ++off) {
        const float w0 = v[base + off];
        const float w1 = v[base + step + off];
        const float w2 = v[base + 2 * step + off];
        const float w3 = v[base + 3 * step + off];
        const float p = w0 - w3, q = w0 + w3;
        const float r = w1 + w2, t = w1 - w2;
        v[base + off] = 0.5f * (p + r);
        v[base + step + off] = 0.5f * (q + t);
        v[base + 2 * step + off] = 0.5f * (q - t);
        v[base + 3 * step + off] = 0.5f * (r - p);
      }
    }
  }
}

inline void range(const std::int8_t* q, const float* scale, std::int64_t r0, std::int64_t r1, std::int64_t k, int g,
                  float* out) {
  std::vector<float> v(static_cast<std::size_t>(g));
  for (std::int64_t r = r0; r < r1; ++r) {
    for (std::int64_t c0 = 0; c0 < k; c0 += g) {
      for (int i = 0; i < g; ++i) v[i] = static_cast<float>(q[r * k + c0 + i]) * scale[r];
      hadamard_group(v.data(), g);
      float* dst = out + r * k + c0;
      for (int i = 0; i < g; ++i) dst[i] = v[i];
    }
  }
}

inline void range_bf16(const std::int8_t* q, const float* scale, std::int64_t r0, std::int64_t r1,
                       std::int64_t k, int g, std::uint16_t* out) {
  std::vector<float> v(static_cast<std::size_t>(g));
  for (std::int64_t r = r0; r < r1; ++r) {
    for (std::int64_t c0 = 0; c0 < k; c0 += g) {
      for (int i = 0; i < g; ++i) v[i] = static_cast<float>(q[r * k + c0 + i]) * scale[r];
      hadamard_group(v.data(), g);
      std::uint16_t* dst = out + r * k + c0;
      for (int i = 0; i < g; ++i) dst[i] = mi::quant::float_to_bf16(v[i]);
    }
  }
}

}  // namespace dequant_detail

// Dequantize straight into bf16 bits, avoiding the fp32 temporary.
inline void dequantize_rows_fast_bf16(const std::int8_t* q, const float* scale, std::int64_t rows, std::int64_t k,
                                      int g, std::uint16_t* out, int requested_threads = -1) {
  if (g == 0) {
    for (std::int64_t i = 0; i < rows * k; ++i) out[i] = mi::quant::float_to_bf16(static_cast<float>(q[i]) * scale[i / k]);
    return;
  }
  const int threads = requested_threads > 0
                          ? requested_threads
                          : std::min<int>(8, static_cast<int>(std::thread::hardware_concurrency()));
  if (threads <= 1 || rows < threads) {
    dequant_detail::range_bf16(q, scale, 0, rows, k, g, out);
    return;
  }
  std::vector<std::thread> workers;
  for (int t = 1; t < threads; ++t) {
    const std::int64_t r0 = rows * t / threads, r1 = rows * (t + 1) / threads;
    workers.emplace_back([=] { dequant_detail::range_bf16(q, scale, r0, r1, k, g, out); });
  }
  dequant_detail::range_bf16(q, scale, 0, rows / threads, k, g, out);
  for (auto& w : workers) w.join();
}

// Bit-comparable to mi::quant::dequantize_rows for a power-of-4 group; parallel over rows.
inline void dequantize_rows_fast(const std::int8_t* q, const float* scale, std::int64_t rows, std::int64_t k, int g,
                                 float* out, int requested_threads = -1) {
  if (g == 0) {
    for (std::int64_t i = 0; i < rows * k; ++i) out[i] = static_cast<float>(q[i]) * scale[i / k];
    return;
  }
  const int threads = requested_threads > 0
                          ? requested_threads
                          : std::min<int>(8, static_cast<int>(std::thread::hardware_concurrency()));
  if (threads <= 1 || rows < threads) {
    dequant_detail::range(q, scale, 0, rows, k, g, out);
    return;
  }
  std::vector<std::thread> workers;
  for (int t = 1; t < threads; ++t) {
    const std::int64_t r0 = rows * t / threads, r1 = rows * (t + 1) / threads;
    workers.emplace_back([=] { dequant_detail::range(q, scale, r0, r1, k, g, out); });
  }
  dequant_detail::range(q, scale, 0, rows / threads, k, g, out);
  for (auto& w : workers) w.join();
}

}  // namespace mi::hip
