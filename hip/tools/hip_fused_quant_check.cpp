// Regression check: the fused SiTU-GLU + ConvRot quantizer must be bit-identical to situ_glu + rotate_quantize.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "runtime.h"

namespace hip_block {
void situ_glu(const std::uint16_t* up_gate, std::int64_t ld, std::uint16_t* out, std::int64_t ldo,
              std::int64_t rows, int f, hipStream_t stream);
}

namespace mi::hip {
void rotate_quantize(const std::uint16_t* x, std::int64_t ldx, std::int64_t rows, std::int64_t k,
                     std::int8_t* q, std::int64_t ldq, float* scale, hipStream_t stream);
void rotate_quantize_situ(const std::uint16_t* up_gate, std::int64_t ld, std::int64_t rows, std::int64_t k,
                          std::int8_t* q, std::int64_t ldq, float* scale, hipStream_t stream);
}

int main() {
  mi::hip::Device device(-1);
  const std::int64_t rows = 3;
  const std::int64_t f = 8192;
  std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * 2 * f);
  unsigned seed = 7;
  for (auto& v : host) { seed = seed * 1664525u + 1013904223u; v = static_cast<std::uint16_t>(seed >> 16); }
  mi::hip::DeviceBuffer dx(host.size() * 2), dffn(static_cast<std::size_t>(rows) * f * 2),
      dq1(static_cast<std::size_t>(rows) * f), dq2(static_cast<std::size_t>(rows) * f),
      ds1(rows * 4), ds2(rows * 4);
  dx.upload(host.data(), host.size() * 2);
  const hipStream_t st = 0;
  hip_block::situ_glu(dx.as<std::uint16_t>(), 2 * f, dffn.as<std::uint16_t>(), f, rows, static_cast<int>(f), st);
  mi::hip::rotate_quantize(dffn.as<std::uint16_t>(), f, rows, f, dq1.as<std::int8_t>(), f, ds1.as<float>(), st);
  mi::hip::rotate_quantize_situ(dx.as<std::uint16_t>(), 2 * f, rows, f, dq2.as<std::int8_t>(), f, ds2.as<float>(), st);
  hipDeviceSynchronize();
  std::vector<std::int8_t> q1(static_cast<std::size_t>(rows) * f), q2(q1.size());
  std::vector<float> s1(rows), s2(rows);
  dq1.download(q1.data(), q1.size()); dq2.download(q2.data(), q2.size());
  ds1.download(s1.data(), s1.size() * 4); ds2.download(s2.data(), s2.size() * 4);
  std::size_t bad = 0; float worst = 0;
  for (std::size_t i = 0; i < q1.size(); ++i) bad += q1[i] != q2[i];
  for (std::size_t i = 0; i < s1.size(); ++i) worst = std::max(worst, std::abs(s1[i] - s2[i]));
  std::printf("fused vs split: bad=%zu/%zu worst_scale=%.9g scale0=%.9g %.9g q0=%d %d\n", bad, q1.size(), worst, s1[0], s2[0], q1[0], q2[0]);
  return bad == 0 ? 0 : 1;
}
