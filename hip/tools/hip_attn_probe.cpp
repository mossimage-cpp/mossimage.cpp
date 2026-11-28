// Fast probe of hip_block::attention against a CPU reference. No model loading.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "hip_block.h"
#include "runtime.h"

using namespace mi::hip;
using bf16 = hip_block::bf16_bits;

static bf16 to_bf(float v) {
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  return static_cast<bf16>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float from_bf(bf16 b) {
  std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}

int main(int argc, char** argv) {
  Device device(-1);
  const int T = argc > 1 ? std::atoi(argv[1]) : 273;
  const int heads = 24, D = 128, K = 3 * 3072;
  const bool packed = argc > 2 ? std::atoi(argv[2]) != 0 : false;
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> qkv(static_cast<std::size_t>(T) * K);
  const char* dump = std::getenv("MI_ATTN_QKV");
  if (dump != nullptr) {
    FILE* f = std::fopen(dump, "rb");
    if (f == nullptr || std::fread(qkv.data(), 2, qkv.size(), f) != qkv.size()) {
      std::printf("cannot read qkv dump\n");
      return 1;
    }
    std::fclose(f);
  } else {
    for (auto& x : qkv) x = to_bf(nd(rng));
  }
  std::vector<std::int32_t> cu{0, T};

  DeviceBuffer dqkv(qkv.size() * 2), dout(static_cast<std::size_t>(T) * 3072 * 2), dcu(cu.size() * 4), dpack;
  dqkv.upload(qkv.data(), qkv.size() * 2);
  dcu.upload(cu.data(), cu.size() * 4);
  hip_block::BlockBuffers bb{};
  bb.qkv = dqkv.as<bf16>();
  bb.cu_seqlens = dcu.as<std::int32_t>();
  bb.batch = 1;
  bb.max_seqlen = T;
  bb.tokens = T;
  if (packed) {
    dpack = DeviceBuffer(static_cast<std::size_t>((T + hip_block::kAttentionStepN) * K * 2));
    bb.pack_attn = dpack.as<bf16>();
    hip_block::pack_attention(bb, 0);
  }

  hip_block::AttentionArgs a{};
  a.q = dqkv.as<bf16>();
  a.k = dqkv.as<bf16>() + 3072;
  a.v = dqkv.as<bf16>() + 6144;
  a.o = dout.as<bf16>();
  a.ldq = a.ldk = a.ldv = K;
  a.ldo = 3072;
  a.hsq = a.hsk = a.hsv = a.hso = D;
  a.cu_q = a.cu_k = dcu.as<std::int32_t>();
  a.batch = 1;
  a.max_seqlen = T;
  a.heads = heads;
  a.scale = static_cast<float>(1.0 / std::sqrt(128.0));
  if (packed) a.packed = bb.pack_attn;
  hip_block::attention(a, 0);

  std::vector<bf16> out(static_cast<std::size_t>(T) * 3072);
  dout.download(out.data(), out.size() * 2);

  const float sc = 1.f / std::sqrt(128.f);
  int bad = 0;
  for (int r = 0; r < T; ++r) {
    for (int h = 0; h < 1; ++h) {
      std::vector<float> Q(D), s(T);
      for (int c = 0; c < D; ++c) Q[c] = from_bf(qkv[static_cast<std::size_t>(r) * K + h * D + c]);
      float mx = -1e30f;
      for (int t = 0; t < T; ++t) {
        float v = 0;
        for (int c = 0; c < D; ++c) v += Q[c] * from_bf(qkv[static_cast<std::size_t>(t) * K + 3072 + h * D + c]);
        s[t] = v * sc;
        mx = std::max(mx, s[t]);
      }
      float sum = 0;
      for (int t = 0; t < T; ++t) s[t] = std::exp(s[t] - mx), sum += s[t];
      std::vector<float> ref(D, 0);
      for (int t = 0; t < T; ++t) {
        for (int c = 0; c < D; ++c) ref[c] += s[t] / sum * from_bf(qkv[static_cast<std::size_t>(t) * K + 6144 + h * D + c]);
      }
      double err = 0;
      for (int c = 0; c < D; ++c) err += std::abs(ref[c] - from_bf(out[static_cast<std::size_t>(r) * 3072 + h * D + c]));
      err /= D;
      if (err > 0.01) ++bad;
      if (r < 20 || r == T - 1) {
        std::printf("row %3d err %.6f ref", r, err);
        for (int c = 0; c < 4; ++c) std::printf(" %.4f", ref[c]);
        std::printf(" got");
        for (int c = 0; c < 4; ++c) std::printf(" %.4f", from_bf(out[static_cast<std::size_t>(r) * 3072 + h * D + c]));
        std::printf("\n");
      }
    }
  }
  std::printf("T=%d packed=%d bad rows=%d/%d\n", T, (int)packed, bad, T);
  return 0;
}
