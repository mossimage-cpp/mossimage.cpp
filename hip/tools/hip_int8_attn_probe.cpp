// Check the int8 attention path against an fp32 CPU reference on a random qkv tensor.
// Usage: hip_int8_attn_probe [tokens]
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "int8_attn.h"
#include "runtime.h"
#include "sage_attn_gfx103x.h"

using namespace mi::hip;
using bf16 = i8attn_bf16;

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
  const int heads = hip_block::kHeads, D = hip_block::kHeadDim, dim = hip_block::kDim;
  const std::int32_t cu[2] = {0, T};
  const std::size_t padded = int8_attn_padded(T, 1);

  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> qkv(static_cast<std::size_t>(T) * 3 * dim);
  for (auto& v : qkv) v = to_bf(nd(rng));

  DeviceBuffer dq(static_cast<std::size_t>(T) * 3 * dim * 2);
  DeviceBuffer dout(static_cast<std::size_t>(T) * dim * 2);
  DeviceBuffer ds(int8_attn_bytes(T, 1));
  auto* base = reinterpret_cast<std::uint8_t*>(ds.data());
  const auto take = [&](std::size_t n) {
    std::uint8_t* r = base;
    base += n;
    return r;
  };
  const std::size_t h = static_cast<std::size_t>(heads) * D;
  Int8AttnBuffers b;
  b.q = reinterpret_cast<std::int8_t*>(take(padded * h));
  b.k = reinterpret_cast<std::int8_t*>(take(padded * h));
  b.v = reinterpret_cast<std::int8_t*>(take(padded * h));
  b.q_scale = reinterpret_cast<float*>(take(padded * heads * sizeof(float)));
  b.k_scale = reinterpret_cast<float*>(take(padded / 16 * heads * sizeof(float)));
  b.v_scale = reinterpret_cast<float*>(take(h * sizeof(float)));
  b.cu = reinterpret_cast<std::int32_t*>(take(2 * sizeof(std::int32_t)));
  b.part = reinterpret_cast<float*>(take(((static_cast<std::size_t>(T) + 255) / 256) * h * sizeof(float)));
  b.batch = 1;
  b.padded = static_cast<int>(padded);

  const hipStream_t st = 0;
  dq.upload(qkv.data(), qkv.size() * 2);
  ds.upload(cu, sizeof(cu), reinterpret_cast<std::uint8_t*>(const_cast<std::int32_t*>(b.cu)) -
                                reinterpret_cast<std::uint8_t*>(ds.data()));
  int8_attn_quantize(dq.as<bf16>(), 3 * dim, b, T, st);
  const bool gfx103x = std::getenv("MI_I8ATTN_BACKEND") != nullptr &&
                       std::string(std::getenv("MI_I8ATTN_BACKEND")) == "gfx103x";
  if (gfx103x) {
    DeviceBuffer dv_fp16(int8_attn_fp16_v_bytes(T, 1));
    gfx103x_v_to_fp16(dq.as<bf16>(), 3 * dim, b.cu, dv_fp16.as<std::uint16_t>(), T, 1, st);
    gfx103x_attn({b.q, b.k, dv_fp16.as<std::uint16_t>(), dout.as<bf16>(), b.q_scale, b.k_scale,
                  static_cast<std::int64_t>(heads) * D, heads, heads, dim, b.cu, 1, T, heads,
                  static_cast<float>(1.0 / std::sqrt(double(D)) * 1.4426950408889634)}, st);
  } else {
    int8_attn_run(b, dout.as<bf16>(), dim, T, static_cast<float>(1.0 / std::sqrt(double(D))), st);
  }
  if (std::getenv("MI_I8ATTN_BENCH")) {
    hipDeviceSynchronize();
    const int iters = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) int8_attn_quantize(dq.as<bf16>(), 3 * dim, b, T, st);
    hipDeviceSynchronize();
    const double qms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      int8_attn_run(b, dout.as<bf16>(), dim, T, static_cast<float>(1.0 / std::sqrt(double(D))), st);
    hipDeviceSynchronize();
    const double ams = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    std::printf("T=%d bench quant %.3f attn %.3f ms/call\n", T, qms / iters, ams / iters);
    return 0;
  }
  std::vector<bf16> got(static_cast<std::size_t>(T) * dim);
  dout.download(got.data(), got.size() * 2);
  if (std::getenv("MI_I8ATTN_DEBUG")) {
    const auto dump_i8 = [&](const char* name, const std::int8_t* p, std::size_t n) {
      std::vector<std::int8_t> h(n);
      hipMemcpy(h.data(), p, n, hipMemcpyDeviceToHost);
      int nz = 0, mx = 0;
      for (auto v : h) {
        nz += v != 0;
        mx = std::max(mx, std::abs(static_cast<int>(v)));
      }
      std::printf("%s nz=%d/%zu max=%d first=%d %d\n", name, nz, n, mx, h[0], h[1]);
    };
    const auto dump_f = [&](const char* name, const float* p, std::size_t n) {
      std::vector<float> h(n);
      hipMemcpy(h.data(), p, n * sizeof(float), hipMemcpyDeviceToHost);
      float mn = 1e30f, mx = -1e30f;
      for (auto v : h) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }
      std::printf("%s min=%g max=%g first=%g %g\n", name, mn, mx, h[0], h[1]);
    };
    dump_i8("q", b.q, padded * h);
    dump_i8("k", b.k, padded * h);
    dump_i8("v", b.v, padded * h);
    dump_f("qscale", b.q_scale, padded * heads);
    dump_f("kscale", b.k_scale, padded / 16 * heads);
    dump_f("vscale", b.v_scale, h);
    std::printf("got %g %g %g %g\n", from_bf(got[0]), from_bf(got[1]), from_bf(got[2]), from_bf(got[3]));
  }

  if (std::getenv("MI_I8ATTN_DEQ")) {
    // CPU attention from the GPU's dequantized buffers: isolates the WMMA kernel from the quantizers.
    std::vector<std::int8_t> qh(padded * h), kh(padded * h), vh(padded * h);
    std::vector<float> qsh(padded * heads), ksh(padded / 16 * heads), vsh(h);
    hipMemcpy(qh.data(), b.q, qh.size(), hipMemcpyDeviceToHost);
    hipMemcpy(kh.data(), b.k, kh.size(), hipMemcpyDeviceToHost);
    hipMemcpy(vh.data(), b.v, vh.size(), hipMemcpyDeviceToHost);
    hipMemcpy(qsh.data(), b.q_scale, qsh.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(ksh.data(), b.k_scale, ksh.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(vsh.data(), b.v_scale, vsh.size() * 4, hipMemcpyDeviceToHost);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    {
      const int cq = static_cast<std::size_t>(0) * heads * D;
      const int ck = static_cast<std::size_t>(0) * heads * D;
      float deq = 0.0f, orig = 0.0f;
      const float qs = qsh[0], ks = ksh[0];
      for (int j = 0; j < D; ++j) {
        deq += static_cast<float>(qh[cq + j]) * qs * static_cast<float>(kh[ck + j]) * ks;
        orig += from_bf(qkv[static_cast<std::size_t>(0) * 3 * dim + j]) *
                from_bf(qkv[static_cast<std::size_t>(0) * 3 * dim + dim + j]);
      }
      std::printf("score(0,0) deq=%g orig=%g qscale=%g kscale=%g\n", deq, orig, qs, ks);
    }
    for (int d = 0; d < 8; ++d) {
      const int row = 0, head = 0;
      float z = 0.0f;
      for (int k = 0; k < T; ++k) {
        float acc = 0.0f;
        const int cq = (static_cast<std::size_t>(row) * heads + head) * D;
        const int ck = (static_cast<std::size_t>(k) * heads + head) * D;
        const float qs = qsh[row * heads + head], ks = ksh[(k / 16) * heads + head];
        for (int j = 0; j < D; ++j) acc += static_cast<float>(qh[cq + j]) * qs *
                                          static_cast<float>(kh[ck + j]) * ks;
        z += std::exp(acc * scale);
      }
      float num = 0.0f;
      for (int k = 0; k < T; ++k) {
        float acc = 0.0f;
        const int cq = (static_cast<std::size_t>(row) * heads + head) * D;
        const int ck = (static_cast<std::size_t>(k) * heads + head) * D;
        const float qs = qsh[row * heads + head], ks = ksh[(k / 16) * heads + head];
        for (int j = 0; j < D; ++j) acc += static_cast<float>(qh[cq + j]) * qs *
                                          static_cast<float>(kh[ck + j]) * ks;
        num += std::exp(acc * scale) / z * static_cast<float>(vh[(head * D + d) * padded + k]) * vsh[head * D + d];
      }
      std::printf("d=%d deq=%g gpu=%g\n", d, num, from_bf(got[head * D + d]));
    }
  }

  // CPU fp32 reference, one head at a time.
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  int bad = 0;
  float worst = 0.0f, esum = 0.0f, rsum = 0.0f;
  for (int head = 0; head < heads; ++head) {
    for (int row = 0; row < T; ++row) {
      float mx = -1e30f;
      std::vector<float> s(T);
      const bf16* q = qkv.data() + static_cast<std::size_t>(row) * 3 * dim + head * D;
      for (int k = 0; k < T; ++k) {
        float acc = 0.0f;
        const bf16* kk = qkv.data() + static_cast<std::size_t>(k) * 3 * dim + dim + head * D;
        for (int d = 0; d < D; ++d) acc += from_bf(q[d]) * from_bf(kk[d]);
        s[k] = acc * scale;
        mx = std::max(mx, s[k]);
      }
      float den = 0.0f;
      for (int k = 0; k < T; ++k) den += std::exp(s[k] - mx);
      for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int k = 0; k < T; ++k) {
          const float w = std::exp(s[k] - mx) / den;
          acc += w * from_bf(qkv[static_cast<std::size_t>(k) * 3 * dim + 2 * dim + head * D + d]);
        }
        const float g = from_bf(got[static_cast<std::size_t>(row) * dim + head * D + d]);
        const float err = std::fabs(acc - g);
        esum += err;
        rsum += std::fabs(acc);
        worst = std::max(worst, err);
        if (err > 0.02f) ++bad;
      }
    }
  }
  std::printf("T=%d bad=%.4f%% worst=%.5f mean_abs_err=%.6f mean_abs_ref=%.6f\n", T,
              100.0 * bad / (T * dim), worst, esum / (T * dim), rsum / (T * dim));
  return bad == 0 ? 0 : 1;
}
