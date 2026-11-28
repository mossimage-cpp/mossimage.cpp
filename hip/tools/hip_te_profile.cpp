// Time TextEncoder construction and encode, with the device's per-op breakdown.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime.h"
#include "text_encoder.h"

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  using namespace mi::hip;
  Device device(-1);
  TextEncoderOptions opt;
  opt.time_ops = true;
  const char* pg = std::getenv("MI_HIP_TE_PACKED_GEMM");
  opt.tiled_gemm = pg == nullptr || std::strcmp(pg, "0") == 0;
  auto t0 = std::chrono::steady_clock::now();
  TextEncoder enc(device, argv[1], opt);
  double ctor = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("ctor: %.2f s\n", ctor);
  t0 = std::chrono::steady_clock::now();
  try {
  auto out = enc.encode({std::vector<std::int32_t>(64, 1)});
  double enc_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const auto& st = enc.last_stats();
  std::printf("encode: %.2f s  total=%.2f embed=%.2f read=%.2f read_wait=%.2f layers=%.0fms weight_wait=%.0fms\n",
              enc_s, st.total_s, st.embed_s, st.read_s, st.read_wait_s, st.layers_ms, st.weight_wait_ms);
  for (const auto& [name, ms] : st.op_ms) std::printf("  %-24s %8.1f ms\n", name.c_str(), ms);
  std::printf("out %zu seqs, first len %zu\n", out.size(), out.empty() ? 0 : out[0].size());
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  return 0;
}
