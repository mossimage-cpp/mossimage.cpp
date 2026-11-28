// <chrono> before the HIP headers: their __noinline__ macro breaks libstdc++'s <format>, which <chrono> includes in C++20.
#include <chrono>

#include "vae.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "safetensors.h"

namespace mi::hip {
namespace {

using hip_vae::Products;

// Activation buffer element types. Both 16-bit modes hold uint16 bits, but they are distinct types so the decode's
// template dispatch can select the right rounding at every kernel boundary. sizeof stays 2 so Act* arithmetic matches
// the device buffers the host allocates.
struct fp16_bits {
  std::uint16_t v;
};
struct bf16_bits {
  std::uint16_t v;
};
static_assert(sizeof(fp16_bits) == 2 && sizeof(bf16_bits) == 2, "16-bit activation tags");

template <class Act>
struct ActFmtOf {
  static constexpr int value = 0;
};
template <>
struct ActFmtOf<fp16_bits> {
  static constexpr int value = 1;
};
template <>
struct ActFmtOf<bf16_bits> {
  static constexpr int value = 2;
};
inline int round_of(Products p) { return p == Products::kFp16 ? 1 : (p == Products::kBf16 ? 2 : 0); }

// The VAE's operand precision, chosen once when a decoder is constructed (gfx1035 defaults to fp16, gfx11 to bf16).
// The static workspace/planner helpers call layout()/attn_pieces() without a decoder, so keep the choice process-global.
int& vae_round_flag() {
  static int round = 0;
  return round;
}
int vae_round() { return vae_round_flag(); }

// 16-bit activation storage: keeps every activation buffer (and the convolutions' inputs and outputs) in the operand's
// 16-bit format, halving their bandwidth. Accumulation, GroupNorm statistics and RGB8 stay fp32. On by default wherever
// a 16-bit operand path is; MI_HIP_VAE_FP16_ACT / MI_HIP_VAE_BF16_ACT = 0 restores fp32 activations.
int& vae_act_fmt_flag() {
  static int fmt = 0;
  return fmt;
}
int vae_act_fmt() { return vae_act_fmt_flag(); }
bool vae_fp16_act() { return vae_act_fmt() != 0; }

constexpr float kResnetEps = 1e-6f;
constexpr float kBatchNormEps = 1e-4f;  // vae.config.batch_norm_eps
constexpr int kUpBlocks = 4, kResnetsPerBlock = 3;
constexpr std::uint64_t kAlign = 256;
// Bytes of one attention slab: its fp32 scores and their packed pieces, [rows, L] each.
constexpr std::uint64_t kSlabBytes = 256ull << 20;

std::uint64_t align_up(std::uint64_t v, std::uint64_t a = kAlign) { return (v + a - 1) / a * a; }

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("hip vae: " + message);
}

int attn_pieces();  // defined below layout(); selected by MI_HIP_VAE_ATTN_PIECES

// Device pointers as byte offsets into the weight buffer until it exists.
struct Conv {
  std::uint64_t w = 0, b = 0;  // packed pieces of the [k, r r c] KRSC weight (conv_out: TF32-rounded fp32 KRSC), bias [k]
  int k = 0, c = 0, r = 0;
};
struct Norm {
  std::uint64_t gamma = 0, beta = 0;
  int c = 0;
};
struct Resnet {
  Norm norm1, norm2;
  Conv conv1, conv2, shortcut;  // shortcut.k == 0 when in == out
  int in = 0, out = 0;
};

}  // namespace

struct VaeDecoder::Weights {
  DeviceBuffer buffer;
  std::uint64_t bn_scale = 0, bn_mean = 0;
  Conv post_quant, conv_in, conv_out, qkv, attn_out;
  Norm attn_norm, norm_out;
  Resnet mid[2];
  Resnet up[kUpBlocks][kResnetsPerBlock];
  Conv upsampler[kUpBlocks - 1];

  const char* base() const { return static_cast<const char*>(buffer.data()); }
  const float* f32(std::uint64_t at) const { return reinterpret_cast<const float*>(base() + at); }
  const std::uint16_t* u16(std::uint64_t at) const { return reinterpret_cast<const std::uint16_t*>(base() + at); }
};

namespace {

// Host image of the weight buffer: tensors at 256-byte offsets.
class Packer {
 public:
  explicit Packer(const st::File& file, Products products) : file_(file), products_(products) {}

  template <class T>
  std::uint64_t add(const std::vector<T>& v) {
    const std::uint64_t at = data_.size();
    data_.resize(align_up(at + v.size() * sizeof(T)));
    std::memcpy(data_.data() + at, v.data(), v.size() * sizeof(T));
    return at;
  }
  std::vector<float> read(const std::string& name, std::initializer_list<std::int64_t> shape) const {
    const st::TensorInfo* t = file_.find(name);
    if (t == nullptr) throw std::runtime_error("hip vae: weight '" + name + "' missing from " + file_.path());
    if (t->shape != std::vector<std::int64_t>(shape)) throw std::runtime_error("hip vae: weight '" + name + "' has an unexpected shape");
    return file_.read_f32(name);
  }
  // A [k, c, r, r] convolution as its packed KRSC pieces, or (narrow) as TF32-rounded fp32 KRSC.
  Conv conv(const std::string& p, int k, int c, int r, bool narrow = false) {
    const std::vector<float> kcrs = read(p + ".weight", {k, c, r, r});
    std::vector<float> krsc(kcrs.size());
    const std::size_t taps = static_cast<std::size_t>(r) * r, cs = static_cast<std::size_t>(c);
    for (std::size_t ki = 0; ki < static_cast<std::size_t>(k); ++ki) {
      for (std::size_t ci = 0; ci < cs; ++ci) {
        for (std::size_t t = 0; t < taps; ++t) krsc[(ki * taps + t) * cs + ci] = kcrs[(ki * cs + ci) * taps + t];
      }
    }
    Conv cv;
    cv.k = k;
    cv.c = c;
    cv.r = r;
    if (narrow) {
      if (products_ == Products::kFp16)
        for (float& v : krsc) v = hip_vae::host_fp16_value(hip_vae::host_fp16_bits(v));
      else if (products_ == Products::kBf16)
        for (float& v : krsc) v = hip_vae::host_bf16_value(hip_vae::host_bf16_bits(v));
      else
        for (float& v : krsc) v = hip_vae::host_tf32(v);
      cv.w = add(krsc);
    } else {
      cv.w = add(packed(krsc, k, static_cast<int>(taps) * c, products_));
    }
    cv.b = add(read(p + ".bias", {k}));
    return cv;
  }
  static std::vector<std::uint16_t> packed(const std::vector<float>& w, int n, int k, Products products) {
    std::vector<std::uint16_t> out(hip_vae::packed_weight_elems(n, k, products));
    hip_vae::pack_weight(w.data(), n, k, products, out.data());
    return out;
  }
  Norm norm(const std::string& p, int c) {
    Norm n;
    n.c = c;
    n.gamma = add(read(p + ".weight", {c}));
    n.beta = add(read(p + ".bias", {c}));
    return n;
  }
  Resnet resnet(const std::string& p, int in, int out) {
    Resnet r;
    r.in = in;
    r.out = out;
    r.norm1 = norm(p + ".norm1", in);
    r.conv1 = conv(p + ".conv1", out, in, 3);
    r.norm2 = norm(p + ".norm2", out);
    r.conv2 = conv(p + ".conv2", out, out, 3);
    if (in != out) r.shortcut = conv(p + ".conv_shortcut", out, in, 1);
    return r;
  }
  const std::vector<char>& data() const { return data_; }

 private:
  const st::File& file_;
  Products products_;
  std::vector<char> data_;
};

// The device memory of one decode.
struct Layout {
  std::uint64_t region = 0;  // bytes per ping-pong region; each holds two halves
  std::uint64_t latents = 0, rgb = 0, cout = 0, aux = 0;  // cout: fp32 [full, 3] before RGB8
  // The mid attention's scratch in the second region, as offsets from its start.
  std::uint64_t xn = 0, qkv = 0, qp = 0, kp = 0, vtp = 0, scores = 0, pp = 0;
  std::int64_t l = 0, l_rows = 0, l_cols = 0, l_k = 0, mid_cols = 0;  // L; Q rows, K rows, PV's K, V^T rows (padded)
  int slab = 0;                                                       // query rows per slab
  std::uint64_t total() const { return 2 * region + latents + rgb + cout + aux; }
};

Layout layout(int c, std::uint32_t h, std::uint32_t w) {
  require(c > 0 && c % 32 == 0, "decoder widths must be multiples of 32");
  require(h > 0 && w > 0, "latent grid must be at least 1x1");
  // The sizes the kernels can address, checked before any product below can overflow.
  const std::string image = "a " + std::to_string(16ull * w) + "x" + std::to_string(16ull * h) + " image";
  require(h <= (1u << 20) && w <= (1u << 20), image + " is past the decoder's limits");
  const std::uint64_t full = 256ull * h * w, mid = 4ull * static_cast<std::uint64_t>(c);
  // up_blocks[3]'s first GroupNorm: 2c channels at full resolution, the largest group and the largest launch (one
  // work-item per 4 channels of a pixel; AMD launches hold fewer than 2^32 work-items).
  require(full * (2ull * static_cast<std::uint64_t>(c) / 32) <= (1ull << 30),
          image + " is past the decoder's limits: a GroupNorm group would hold more than 2^30 elements");
  require(full * (2ull * static_cast<std::uint64_t>(c) / 4) < (1ull << 32),
          image + " is past the decoder's limits: a GroupNorm launch would need 2^32 work-items or more");
  // 16-bit activation storage halves the ping-pong regions and the attention's normalized/linear buffers. The scores,
  // softmax pieces and GroupNorm accumulators keep their fp32/packed-operand layouts.
  const std::uint64_t esz = vae_fp16_act() ? 2ull : 4ull;
  Layout l;
  l.l = 4ll * h * w;
  l.l_rows = hip_vae::round_up(l.l, hip_vae::kTileM);
  l.l_cols = hip_vae::round_up(l.l, hip_vae::kTileN);
  l.l_k = hip_vae::round_up(l.l, 16);
  l.mid_cols = hip_vae::round_up(static_cast<std::int64_t>(mid), hip_vae::kTileN);
  const auto lk = static_cast<std::uint64_t>(l.l_k), ll = static_cast<std::uint64_t>(l.l);
  const int pcs = attn_pieces();
  const std::uint64_t per_row = lk * (4 + static_cast<std::uint64_t>(pcs) * 2);
  l.slab = static_cast<int>(std::min<std::uint64_t>(static_cast<std::uint64_t>(l.l_rows),
                                                    std::max<std::uint64_t>(32, kSlabBytes / per_row / 32 * 32)));
  const auto slab = static_cast<std::uint64_t>(l.slab);
  std::uint64_t at = 0;
  const auto take = [&](std::uint64_t bytes) {
    const std::uint64_t r = at;
    at += align_up(bytes);
    return r;
  };
  l.xn = take(ll * mid * esz);
  l.qkv = take(ll * 3 * mid * esz);
  l.qp = take(static_cast<std::uint64_t>(pcs) * l.l_rows * mid * 2);
  l.kp = take(static_cast<std::uint64_t>(pcs) * l.l_cols * mid * 2);
  l.vtp = take(static_cast<std::uint64_t>(pcs) * l.mid_cols * lk * 2);
  l.scores = take(slab * lk * 4);
  l.pp = take(static_cast<std::uint64_t>(pcs) * slab * lk * 2);
  const std::uint64_t tail = 2ull * static_cast<std::uint64_t>(c) * full * esz;  // up_blocks[2]'s upsampler output
  l.region = align_up(std::max(tail, at), 2 * kAlign);
  l.latents = align_up(128ull * h * w * 4);
  l.rgb = align_up(full * 3);
  l.cout = align_up(full * 3 * 4);
  l.aux = align_up(hip_vae::gn_accum_bytes()) + kAlign;  // GroupNorm accumulators, then {mean, rstd} per group
  return l;
}

double gib(std::uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0); }

// One-product modes (fp16 on gfx103x, bf16 on gfx11/gfx12) use one rounded operand; the TF32 path packs 2 bf16 pieces
// by default (3 instead of 6 GEMM products). MI_HIP_VAE_ATTN_PIECES=3 restores the exact 3-piece path in the NVIDIA
// check build; on AMD devices a one-product mode is always active.
int attn_pieces() {
  if (vae_round() != 0) return 1;
  if (const char* s = std::getenv("MI_HIP_VAE_ATTN_PIECES")) {
    const int v = std::atoi(s);
    if (v == 2 || v == 3) return v;
  }
  return 2;
}

// Records each op between two timing events when profiling.
class Ops {
 public:
  Ops(hipStream_t stream, bool profile) : stream_(stream), profile_(profile) {}
  template <class F>
  void run(const std::string& name, F&& op) {
    if (!profile_) {
      op();
      return;
    }
    auto start = std::make_unique<Event>(true), stop = std::make_unique<Event>(true);
    start->record(stream_);
    op();
    stop->record(stream_);
    timed_.push_back({name, std::move(start), std::move(stop)});
  }
  // After the stream finished: milliseconds per name, in order of first use.
  std::vector<std::pair<std::string, double>> profile() const {
    std::vector<std::pair<std::string, double>> out;
    for (const Timed& t : timed_) {
      const double ms = t.stop->elapsed_ms(*t.start);
      auto it = std::find_if(out.begin(), out.end(), [&](const auto& e) { return e.first == t.name; });
      if (it == out.end()) {
        out.emplace_back(t.name, ms);
      } else {
        it->second += ms;
      }
    }
    return out;
  }

 private:
  struct Timed {
    std::string name;
    std::unique_ptr<Event> start, stop;
  };
  hipStream_t stream_;
  bool profile_;
  std::vector<Timed> timed_;
};

std::string dims(int h, int w) { return " @" + std::to_string(h) + "x" + std::to_string(w); }

template <class Act>
class Run {
 public:
  static constexpr int kActFmt = ActFmtOf<Act>::value;
  Run(const VaeDecoder::Weights& w, Ops& ops, hipStream_t stream, Products products, void* accum, float* stats)
      : w_(w), ops_(ops), stream_(stream), products_(products), accum_(accum), stats_(stats) {}

  void conv(const Conv& cv, const Act* x, int h, int w, Act* y, const Act* residual = nullptr, bool upsample = false) {
    hip_vae::ConvArgs a;
    a.x = reinterpret_cast<const float*>(x);
    a.w = w_.u16(cv.w);
    a.bias = w_.f32(cv.b);
    a.residual = reinterpret_cast<const float*>(residual);
    a.out = reinterpret_cast<float*>(y);
    a.x_fmt = kActFmt;
    a.out_fmt = kActFmt;
    a.ldo = cv.k;
    a.h = h;
    a.w_out = w;
    a.c = cv.c;
    a.n = cv.k;
    a.taps = cv.r * cv.r;
    a.upsample = upsample;
    const std::string name = "conv " + std::to_string(cv.r) + "x" + std::to_string(cv.r) + " " + std::to_string(cv.c) + "->" +
                             std::to_string(cv.k) + (upsample ? " upsampled" : "") + dims(h, w);
    ops_.run(name, [&] { hip_vae::conv(a, products_, stream_); });
  }
  void norm(const Norm& n, const Act* x, std::int64_t pixels, Act* y, bool silu) {
    const std::string shape = " " + std::to_string(n.c) + " ch, " + std::to_string(pixels) + " px";
    ops_.run("group_norm stats" + shape, [&] {
      hip_vae::group_norm_stats(reinterpret_cast<const float*>(x), pixels, n.c, kResnetEps, accum_, stats_, stream_, kActFmt);
    });
    ops_.run(std::string(silu ? "group_norm+silu apply" : "group_norm apply") + shape, [&] {
      hip_vae::group_norm_apply(reinterpret_cast<const float*>(x), reinterpret_cast<float*>(y), pixels, n.c, stats_,
                                w_.f32(n.gamma), w_.f32(n.beta), silu, stream_, kActFmt, kActFmt);
    });
  }
  // ResnetBlock2D: returns the buffer holding the output, x or a.
  Act* resnet(const Resnet& r, Act* x, Act* a, Act* b, int h, int w) {
    const std::int64_t px = static_cast<std::int64_t>(h) * w;
    if (r.shortcut.k != 0) {
      conv(r.shortcut, x, h, w, a);
      norm(r.norm1, x, px, x, true);
      conv(r.conv1, x, h, w, b);
      norm(r.norm2, b, px, b, true);
      conv(r.conv2, b, h, w, a, a);
      return a;
    }
    norm(r.norm1, x, px, a, true);
    conv(r.conv1, a, h, w, b);
    norm(r.norm2, b, px, b, true);
    conv(r.conv2, b, h, w, x, x);
    return x;
  }
  // x [L, mid] += to_out(SDPA(to_q/k/v(GN(x)))), scratch in ws (the second region).
  void attention(Act* x, char* ws, const Layout& l, int mid) {
    const int len = static_cast<int>(l.l);
    Act* xn = reinterpret_cast<Act*>(ws + l.xn);
    Act* qkv = reinterpret_cast<Act*>(ws + l.qkv);
    auto* qp = reinterpret_cast<std::uint16_t*>(ws + l.qp);
    auto* kp = reinterpret_cast<std::uint16_t*>(ws + l.kp);
    auto* vtp = reinterpret_cast<std::uint16_t*>(ws + l.vtp);
    float* scores = reinterpret_cast<float*>(ws + l.scores);
    auto* pp = reinterpret_cast<std::uint16_t*>(ws + l.pp);
    norm(w_.attn_norm, x, l.l, xn, false);
    linear("attention qkv", w_.qkv, xn, len, qkv, nullptr);
    const int pcs = attn_pieces();
    const int round = round_of(products_);
    // A stride is the size of *one* packed piece; the buffer holds `pcs` pieces at that stride.
    const std::int64_t q_stride = l.l_rows * mid, k_stride = l.l_cols * mid, v_stride = l.mid_cols * l.l_k;
    ops_.run("attention split q, k, v", [&] {
      hip_vae::split_pack(reinterpret_cast<const float*>(qkv), 3 * mid, false, len, mid, qp, static_cast<int>(l.l_rows), mid,
                          stream_, pcs, kActFmt, round);
      hip_vae::split_pack(reinterpret_cast<const float*>(qkv + mid), 3 * mid, false, len, mid, kp, static_cast<int>(l.l_cols),
                          mid, stream_, pcs, kActFmt, round);
      hip_vae::split_pack(reinterpret_cast<const float*>(qkv + 2 * mid), 3 * mid, true, mid, len, vtp,
                          static_cast<int>(l.mid_cols), static_cast<int>(l.l_k), stream_, pcs, kActFmt, round);
    });
    // SDPA's scale as the engine forms it: 1 / sqrt(E) in fp32.
    const float scale = 1.0f / std::sqrt(static_cast<float>(mid));
    Act* o = xn;  // the GroupNorm output is dead after the projections
    const std::int64_t p_stride = static_cast<std::int64_t>(l.slab) * l.l_k;
    for (int r0 = 0; r0 < len; r0 += l.slab) {
      const int rows = std::min(l.slab, len - r0);
      ops_.run("attention scores", [&] {
        hip_vae::piece_gemm(qp + static_cast<std::int64_t>(r0) * mid, q_stride, kp, k_stride, mid, scores, l.l_k, rows, len,
                            stream_, pcs, 0, round);
      });
      ops_.run("attention softmax", [&] {
        hip_vae::softmax_pack(scores, l.l_k, rows, len, scale, pp, p_stride, static_cast<int>(l.l_k), stream_, pcs, round);
      });
      ops_.run("attention P V", [&] {
        hip_vae::piece_gemm(pp, p_stride, vtp, v_stride, static_cast<int>(l.l_k),
                            reinterpret_cast<float*>(o + static_cast<std::int64_t>(r0) * mid), mid, rows,
                            mid, stream_, pcs, kActFmt, round);
      });
    }
    // The reference order: round(to_out(o) + bias), then + residual.
    linear("attention out", w_.attn_out, o, len, x, x);
  }

 private:
  // out [rows, k] = (x [rows, c] · Wᵀ + b) + residual, exact fp32 products.
  void linear(const char* name, const Conv& cv, const Act* x, int rows, Act* out, const Act* residual) {
    hip_vae::ConvArgs a;
    a.x = reinterpret_cast<const float*>(x);
    a.w = w_.u16(cv.w);
    a.bias = w_.f32(cv.b);
    a.residual = reinterpret_cast<const float*>(residual);
    a.out = reinterpret_cast<float*>(out);
    a.x_fmt = kActFmt;
    a.out_fmt = kActFmt;
    a.ldo = cv.k;
    a.h = 1;
    a.w_out = rows;
    a.c = cv.c;
    a.n = cv.k;
    a.taps = 1;
    const Products p = round_of(products_) != 0 ? products_ : Products::kFp32;
    ops_.run(name, [&] { hip_vae::conv(a, p, stream_); });
  }

  const VaeDecoder::Weights& w_;
  Ops& ops_;
  hipStream_t stream_;
  Products products_;
  void* accum_;
  float* stats_;
};

// The head is global (the mid attention sees the whole image), so both decodes run it exactly once, untiled. `p`
// receives the head's output (the first checkpoint in the tiled decode); `scratch` is the second ping-pong region,
// reused by the attention. The op order and every rounding point are those of the full decode().
template <class Act>
void run_head_ops(const VaeDecoder::Weights& wt, Ops& ops, Products products, hipStream_t stream, const float* z,
                  const Layout& l, Act* p, char* scratch, void* accum, float* stats, int mid, int hh, int ww,
                  std::uint32_t h, std::uint32_t w) {
  Run<Act> run(wt, ops, stream, products, accum, stats);
  const std::uint64_t half = l.region / 2;
  Act* unpacked = reinterpret_cast<Act*>(scratch);  // dead after conv_in; reused as the resnet's scratch
  Act* pq = reinterpret_cast<Act*>(scratch + half);
  ops.run("latent unpack", [&] {
    hip_vae::latent_unpack(z, wt.f32(wt.bn_scale), wt.f32(wt.bn_mean), reinterpret_cast<float*>(unpacked),
                           static_cast<int>(h), static_cast<int>(w), stream, Run<Act>::kActFmt);
  });
  run.conv(wt.post_quant, unpacked, hh, ww, pq);
  run.conv(wt.conv_in, pq, hh, ww, p);
  run.resnet(wt.mid[0], p, unpacked, pq, hh, ww);
  run.attention(p, scratch, l, mid);
}

// The full decode's device work, instantiated for fp32, fp16 or bf16 activation storage. The caller has already selected
// the device, checked the memory budget and started the wall clock.
template <class Act>
VaeDecodeStats decode_impl(const VaeDecoder::Weights& wt, Products products, bool profile, int c, const float* latents,
                           std::uint32_t h, std::uint32_t w, std::vector<std::uint8_t>& rgb, std::vector<float>* image) {
  constexpr int kActFmt = ActFmtOf<Act>::value;
  const int mid = 4 * c;
  const Layout l = layout(c, h, w);
  const int hh = static_cast<int>(2 * h), ww = static_cast<int>(2 * w), H = static_cast<int>(16 * h), W = static_cast<int>(16 * w);
  Stream stream;
  // Keep cout last: inserting a buffer before aux shifts aux's address and, on gfx1035, changes the decoded bytes by
  // a few levels. Allocating it after aux preserves the byte-identical decode of the fp32-activation path.
  DeviceBuffer r0(l.region), r1(l.region), z(l.latents), out(l.rgb), aux(l.aux), cout(l.cout);
  struct Drain {
    hipStream_t s;
    ~Drain() { (void)hipStreamSynchronize(s); }
  } drain{stream.get()};
  copy_async(z.data(), latents, 128ull * h * w * 4, stream.get());
  void* accum = aux.data();
  float* stats = reinterpret_cast<float*>(static_cast<char*>(aux.data()) + align_up(hip_vae::gn_accum_bytes()));
  const std::uint64_t half = l.region / 2;
  const auto at = [](const DeviceBuffer& b, std::uint64_t off = 0) {
    return reinterpret_cast<Act*>(static_cast<char*>(b.data()) + off);
  };

  Ops ops(stream.get(), profile);
  Run<Act> run(wt, ops, stream.get(), products, accum, stats);
  Act* p = at(r0);
  Act* q = at(r1);

  // Head at H/8: unpack, post_quant_conv, conv_in, mid resnet 0, mid attention.
  Act* unpacked = q;
  Act* pq = at(r1, half);
  ops.run("latent unpack", [&] {
    hip_vae::latent_unpack(z.as<float>(), wt.f32(wt.bn_scale), wt.f32(wt.bn_mean), reinterpret_cast<float*>(unpacked),
                           static_cast<int>(h), static_cast<int>(w), stream.get(), kActFmt);
  });
  run.conv(wt.post_quant, unpacked, hh, ww, pq);
  run.conv(wt.conv_in, pq, hh, ww, p);
  run.resnet(wt.mid[0], p, q, at(r1, half), hh, ww);
  run.attention(p, static_cast<char*>(r1.data()), l, mid);

  // Tail: mid resnet 1, the up blocks, norm_out + SiLU, conv_out, RGB8. conv_out's narrow output stays fp32.
  int ch = hh, cw = ww;
  run.resnet(wt.mid[1], p, q, at(r1, half), ch, cw);
  for (int b = 0; b < kUpBlocks; ++b) {
    Act* x = p;
    for (int r = 0; r < kResnetsPerBlock; ++r) {
      Act* other = x == at(r0) ? at(r1) : at(r0);
      x = run.resnet(wt.up[b][r], x, other, reinterpret_cast<Act*>(reinterpret_cast<char*>(other) + half), ch, cw);
    }
    if (b + 1 < kUpBlocks) {
      Act* y = x == at(r0) ? at(r1) : at(r0);
      ch *= 2;
      cw *= 2;
      run.conv(wt.upsampler[b], x, ch, cw, y, nullptr, true);
      x = y;
    }
    p = x;
    q = x == at(r0) ? at(r1) : at(r0);
  }
  run.norm(wt.norm_out, p, static_cast<std::int64_t>(ch) * cw, p, true);
  // conv_out's narrow result stays fp32 in its own buffer; the activations may be fp16.
  float* rgb_src = cout.as<float>();
  ops.run("conv_out 3x3" + dims(ch, cw), [&] {
    hip_vae::conv_narrow(reinterpret_cast<const float*>(p), wt.f32(wt.conv_out.w), wt.f32(wt.conv_out.b),
                         rgb_src, ch, cw, c, 3, stream.get(), round_of(products), kActFmt);
  });
  const std::int64_t n = static_cast<std::int64_t>(H) * W * 3;
  ops.run("rgb8", [&] { hip_vae::rgb8(rgb_src, out.as<std::uint8_t>(), n, stream.get()); });
  stream.synchronize();

  rgb.resize(static_cast<std::size_t>(n));
  out.download(rgb.data(), static_cast<std::size_t>(n));
  if (image != nullptr) {
    image->resize(static_cast<std::size_t>(n));
    check(hipMemcpy(image->data(), rgb_src, static_cast<std::size_t>(n) * 4, hipMemcpyDeviceToHost), "hipMemcpy of the image");
  }
  VaeDecodeStats st;
  st.device_bytes = l.total();
  if (profile) st.profile = ops.profile();
  return st;
}

}  // namespace

VaeDecoder::VaeDecoder(Device& device, const std::filesystem::path& path) : device_(device), weights_(std::make_unique<Weights>()) {
  device_.make_current();
  const st::File file(path);
  const st::TensorInfo* conv_in = file.find("decoder.conv_in.weight");
  if (conv_in == nullptr || conv_in->shape.size() != 4 || conv_in->shape[0] % 4 != 0 || conv_in->shape[1] != 32) {
    throw std::runtime_error("hip vae: " + file.path() + " has no FLUX.2 decoder.conv_in.weight [4c, 32, 3, 3]");
  }
  base_ = static_cast<int>(conv_in->shape[0] / 4);
  require(base_ % 32 == 0, "decoder widths must be multiples of 32");
  const int c = base_, mid = 4 * c;
  Weights& w = *weights_;
  // gfx1035 has no bf16 arithmetic: the TF32 path would need four fp16 dot products per MAC, so it defaults to one
  // fp16-rounded operand (ComfyUI's --fp16-vae). gfx11/gfx12 have native bf16 WMMA, so they default to one bf16 operand
  // instead of four bf16 pieces; the leaf is the raw __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32, not rocWMMA.
  // MI_HIP_VAE_FP16=0 / MI_HIP_VAE_BF16=0 restore the exact TF32 path. Ask this decoder's device, not whichever device
  // happens to be current (the process may hold several).
  const std::string arch = device_.info().arch;
  const bool rdna2 = arch.compare(0, 6, "gfx103") == 0;
  const bool matrix = arch.compare(0, 5, "gfx11") == 0 || arch.compare(0, 5, "gfx12") == 0;
  const char* fp16_env = std::getenv("MI_HIP_VAE_FP16");
  const char* bf16_env = std::getenv("MI_HIP_VAE_BF16");
  const auto off = [](const char* s) { return s != nullptr && std::strcmp(s, "0") == 0; };
  const bool fp16 = !off(fp16_env) && (fp16_env != nullptr || rdna2);
  const bool bf16 = !fp16 && !off(bf16_env) && (bf16_env != nullptr || matrix);
  if (fp16 && !rdna2) throw std::runtime_error("hip vae: MI_HIP_VAE_FP16 needs a gfx103x (RDNA2) device");
  if (bf16 && !matrix) throw std::runtime_error("hip vae: MI_HIP_VAE_BF16 needs a gfx11/gfx12 device");
  const Products products = fp16 ? Products::kFp16 : (bf16 ? Products::kBf16 : Products::kTf32);
  const int round = round_of(products);
  const char* act_env = std::getenv(round == 1 ? "MI_HIP_VAE_FP16_ACT" : "MI_HIP_VAE_BF16_ACT");
  const int act = round != 0 && !off(act_env) ? round : 0;
  act_fmt_ = act;
  vae_round_flag() = round;
  vae_act_fmt_flag() = act;
  conv_products_ = products;
  Packer pk(file, products);

  const std::vector<float> mean = pk.read("bn.running_mean", {128}), var = pk.read("bn.running_var", {128});
  std::vector<float> scale(128);
  for (std::size_t i = 0; i < 128; ++i) scale[i] = std::sqrt(var[i] + kBatchNormEps);  // correctly rounded, as torch
  w.bn_scale = pk.add(scale);
  w.bn_mean = pk.add(mean);
  w.post_quant = pk.conv("post_quant_conv", 32, 32, 1);
  w.conv_in = pk.conv("decoder.conv_in", mid, 32, 3);
  for (int i = 0; i < 2; ++i) w.mid[i] = pk.resnet("decoder.mid_block.resnets." + std::to_string(i), mid, mid);
  const std::string a = "decoder.mid_block.attentions.0.";
  w.attn_norm = pk.norm(a + "group_norm", mid);
  {
    // to_q, to_k, to_v as one [3 mid, mid] linear: the same products and sums per output as three.
    std::vector<float> wqkv, bqkv;
    for (const char* p : {"to_q", "to_k", "to_v"}) {
      const std::vector<float> wt = pk.read(a + p + ".weight", {mid, mid}), bt = pk.read(a + p + ".bias", {mid});
      wqkv.insert(wqkv.end(), wt.begin(), wt.end());
      bqkv.insert(bqkv.end(), bt.begin(), bt.end());
    }
    const Products ap = round != 0 ? products : Products::kFp32;
    w.qkv = Conv{pk.add(Packer::packed(wqkv, 3 * mid, mid, ap)), pk.add(bqkv), 3 * mid, mid, 1};
    const std::vector<float> wo = pk.read(a + "to_out.0.weight", {mid, mid});
    w.attn_out = Conv{pk.add(Packer::packed(wo, mid, mid, ap)), pk.add(pk.read(a + "to_out.0.bias", {mid})), mid, mid, 1};
  }
  int prev = mid;
  for (int b = 0; b < kUpBlocks; ++b) {
    const int out = c * (b < 2 ? 4 : b == 2 ? 2 : 1);
    const std::string p = "decoder.up_blocks." + std::to_string(b);
    for (int r = 0; r < kResnetsPerBlock; ++r) w.up[b][r] = pk.resnet(p + ".resnets." + std::to_string(r), r == 0 ? prev : out, out);
    if (b + 1 < kUpBlocks) w.upsampler[b] = pk.conv(p + ".upsamplers.0.conv", out, out, 3);
    prev = out;
  }
  w.norm_out = pk.norm("decoder.conv_norm_out", c);
  w.conv_out = pk.conv("decoder.conv_out", 3, c, 3, true);

  const std::vector<char>& data = pk.data();
  w.buffer = DeviceBuffer(data.size());
  w.buffer.upload(data.data(), data.size());
  // That hipMemcpy from pageable memory can return before its DMA lands, and decode()'s stream does not wait for it.
  device_.synchronize();
}

VaeDecoder::~VaeDecoder() = default;

std::uint64_t VaeDecoder::weight_bytes() const { return weights_->buffer.bytes(); }

std::uint64_t VaeDecoder::workspace_bytes(int base_channels, std::uint32_t h, std::uint32_t w) {
  return layout(base_channels, h, w).total();
}

void VaeDecoder::require_fits(std::uint32_t h, std::uint32_t w) const {
  const std::uint64_t need = workspace_bytes(h, w), free = device_.free_memory();
  if (need > free) {
    char msg[256];
    std::snprintf(msg, sizeof msg, "the VAE decode of a %ux%u image needs %.2f GiB of device memory, %s has %.2f GiB free",
                  16 * w, 16 * h, gib(need), device_.info().name.c_str(), gib(free));
    throw std::runtime_error(msg);
  }
}

void VaeDecoder::set_conv_products(Products products) {
  require(products != Products::kFp32, "the convolutions' weights are packed for FP16/BF16/TF32 products");
  require(round_of(conv_products_) == round_of(products),
          "different operand formats cannot be mixed; set MI_HIP_VAE_FP16/BF16 before constructing the decoder");
  conv_products_ = products;
}

VaeDecodeStats VaeDecoder::decode(const float* latents, std::uint32_t h, std::uint32_t w, std::vector<std::uint8_t>& rgb,
                                  std::vector<float>* image) {
  const auto t0 = std::chrono::steady_clock::now();
  device_.make_current();
  // The layout/planner helpers are static and read process-global mode flags. Reassert this decoder's mode so using two
  // decoders in one process cannot make one decode with the other's activation storage or attention pieces.
  vae_round_flag() = round_of(conv_products_);
  vae_act_fmt_flag() = act_fmt_;
  require_fits(h, w);
  VaeDecodeStats st;
  if (act_fmt_ == 2)
    st = decode_impl<bf16_bits>(*weights_, conv_products_, profile_, base_, latents, h, w, rgb, image);
  else if (act_fmt_ == 1)
    st = decode_impl<fp16_bits>(*weights_, conv_products_, profile_, base_, latents, h, w, rgb, image);
  else
    st = decode_impl<float>(*weights_, conv_products_, profile_, base_, latents, h, w, rgb, image);
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return st;
}

// ---- Exact band-tiled decode: planner -----------------------------------------------------------------------------------

namespace {

using hip_vae::kGroups;

constexpr int kTailUnits = 17;         // units of the tail, see VaeTilePlan
constexpr int kMaxSegmentUnits = 6;    // longer segments only add recomputation
constexpr int kStatsSlots = 2 * kTailUnits;
constexpr double kWorkPerByte = 29.0;  // relative cost of an elementwise byte read versus one convolution MAC, for ranking plans

enum class UnitKind : std::uint8_t { kResnet, kUpsample, kOut };
struct UnitShape {
  UnitKind kind;
  int level, in_ch, out_ch;  // an upsampler reads `level` and writes level + 1
};
using Units = std::array<UnitShape, kTailUnits>;

constexpr Units make_units(int base) {
  Units u{};
  int n = 0, ch = 4 * base;
  u[n++] = {UnitKind::kResnet, 0, ch, ch};  // mid_block.resnets[1]
  for (int b = 0; b < 4; ++b) {
    const int out = base * (b < 2 ? 4 : b == 2 ? 2 : 1);
    for (int r = 0; r < kResnetsPerBlock; ++r) {
      u[n++] = {UnitKind::kResnet, b, ch, out};
      ch = out;
    }
    if (b < 3) u[n++] = {UnitKind::kUpsample, b, ch, ch};
  }
  u[n++] = {UnitKind::kOut, 3, ch, 3};
  return u;
}
static_assert(make_units(96)[kTailUnits - 1].kind == UnitKind::kOut && make_units(96)[kTailUnits - 1].in_ch == 96 &&
                  make_units(128)[kTailUnits - 1].in_ch == 128,
              "tail unit table");

constexpr int out_level(const UnitShape& u) { return u.kind == UnitKind::kUpsample ? u.level + 1 : u.level; }

enum class OpKind : std::uint8_t { kConv3, kConv1, kNorm, kUp };
struct Op {
  OpKind kind;
  int level, in_ch, out_ch;  // kUp writes level + 1
};

// A shortcut resnet computes its 1x1 shortcut first, matching Run::resnet().
void append_ops(const UnitShape& u, std::vector<Op>& ops) {
  const int l = u.level;
  switch (u.kind) {
    case UnitKind::kResnet:
      if (u.in_ch != u.out_ch) ops.push_back({OpKind::kConv1, l, u.in_ch, u.out_ch});
      ops.push_back({OpKind::kNorm, l, u.in_ch, u.in_ch});
      ops.push_back({OpKind::kConv3, l, u.in_ch, u.out_ch});
      ops.push_back({OpKind::kNorm, l, u.out_ch, u.out_ch});
      ops.push_back({OpKind::kConv3, l, u.out_ch, u.out_ch});
      break;
    case UnitKind::kUpsample:
      ops.push_back({OpKind::kUp, l, u.in_ch, u.in_ch});
      ops.push_back({OpKind::kConv3, l + 1, u.in_ch, u.in_ch});
      break;
    case UnitKind::kOut:
      ops.push_back({OpKind::kNorm, l, u.in_ch, u.in_ch});
      ops.push_back({OpKind::kConv3, l, u.in_ch, u.out_ch});
      break;
  }
}

struct Geometry {
  std::int64_t h8, w8;  // H/8 x W/8 (twice the latent grid)
  std::size_t esize = 4;
  double work_per_byte = kWorkPerByte;
  std::int64_t rows(int level) const { return h8 << level; }
  std::int64_t cols(int level) const { return w8 << level; }
  std::size_t row_bytes(int level, int ch) const { return static_cast<std::size_t>(cols(level) * ch) * esize; }
};

double op_work(const Op& o, const Geometry& g) {
  const double bytes = g.work_per_byte * static_cast<double>(g.esize);
  switch (o.kind) {
    case OpKind::kConv3: return 9.0 * o.in_ch * o.out_ch;
    case OpKind::kConv1: return 1.0 * o.in_ch * o.out_ch;
    case OpKind::kNorm: return bytes * 2.0 * o.in_ch;
    case OpKind::kUp: return bytes * 1.25 * o.in_ch;
  }
  return 0;
}
double read_work(const Geometry& g, int level, int ch, std::int64_t rows) {
  return g.work_per_byte * static_cast<double>(g.row_bytes(level, ch)) * static_cast<double>(rows);
}
int op_compute_level(const Op& o) { return o.kind == OpKind::kUp ? o.level + 1 : o.level; }

// A pass target: a GroupNorm's input (statistics pass) or the segment output (output pass).
struct Target {
  int end = 0;
  int level = 0, ch = 0;
  // Input-level rows a band needs beyond its owned rows (exact on the band grid, then clipped to the image).
  std::int64_t halo_top = 0, halo_bot = 0;
  double work_per_row = 0;
  bool direct = false;  // the GN input is the checkpoint itself (only 1x1 shortcuts precede it)
};

struct SegmentInfo {
  int first = 0, last = 0;
  int in_level = 0, out_level = 0, in_ch = 0, out_ch = 0;  // out_ch = 0: RGB output
  std::vector<Op> ops;
  std::vector<Target> targets;  // the GroupNorms in order, then the output
  double buf_per_row = 0;       // largest tile tensor, bytes per input-level row of the band
  int groupnorms() const { return static_cast<int>(targets.size()) - 1; }
};

SegmentInfo segment_info(const Geometry& g, const Units& units, int first, int last) {
  SegmentInfo si;
  si.first = first;
  si.last = last;
  si.in_level = units[static_cast<std::size_t>(first)].level;
  si.out_level = out_level(units[static_cast<std::size_t>(last - 1)]);
  si.in_ch = units[static_cast<std::size_t>(first)].in_ch;
  si.out_ch = units[static_cast<std::size_t>(last - 1)].kind == UnitKind::kOut
                  ? 0
                  : units[static_cast<std::size_t>(last - 1)].out_ch;
  for (int u = first; u < last; ++u) append_ops(units[static_cast<std::size_t>(u)], si.ops);
  const auto per_row = [&](int level, int ch) {  // tile tensor bytes per input-level row
    return static_cast<double>(g.row_bytes(level, ch)) * static_cast<double>(std::int64_t{1} << (level - si.in_level));
  };
  si.buf_per_row = per_row(si.in_level, si.in_ch);
  for (const Op& o : si.ops) {
    si.buf_per_row = std::max(si.buf_per_row, per_row(o.level, o.in_ch));
    si.buf_per_row = std::max(si.buf_per_row, per_row(op_compute_level(o), std::max(o.out_ch, 4)));
  }
  const auto make_target = [&](int end, int level, int ch) {
    Target t;
    t.end = end;
    t.level = level;
    t.ch = ch;
    constexpr std::int64_t kAligned = std::int64_t{1} << 24;  // a row index aligned to every band grid
    std::int64_t a = kAligned, b = kAligned;
    t.direct = true;
    for (int k = end - 1; k >= 0; --k) {
      const Op& o = si.ops[static_cast<std::size_t>(k)];
      if (o.kind == OpKind::kConv3) --a, ++b;
      if (o.kind == OpKind::kUp) a >>= 1, b = (b + 1) >> 1;
      if (o.kind != OpKind::kConv1) t.direct = false;
    }
    t.halo_top = (kAligned >> (level - si.in_level)) - a;
    t.halo_bot = b - (kAligned >> (level - si.in_level));
    for (int k = 0; k < end; ++k) {
      const Op& o = si.ops[static_cast<std::size_t>(k)];
      const int cl = op_compute_level(o);
      t.work_per_row += op_work(o, g) * static_cast<double>(g.cols(cl)) *
                        static_cast<double>(std::int64_t{1} << (cl - si.in_level));
    }
    return t;
  };
  int level = si.in_level;
  for (std::size_t k = 0; k < si.ops.size(); ++k) {
    const Op& o = si.ops[k];
    if (o.kind == OpKind::kNorm) si.targets.push_back(make_target(static_cast<int>(k), level, o.in_ch));
    level = op_compute_level(o);
  }
  si.targets.push_back(make_target(static_cast<int>(si.ops.size()), si.out_level, si.out_ch));
  return si;
}

struct Band {
  std::int64_t o0, o1;  // owned rows at the segment output level
};
std::int64_t band_count(const Geometry& g, const SegmentInfo& si, std::int64_t band_rows) {
  return (g.rows(si.out_level) + band_rows - 1) / band_rows;
}
Band band(const Geometry& g, const SegmentInfo& si, std::int64_t band_rows, std::int64_t k) {
  return {k * band_rows, std::min(g.rows(si.out_level), (k + 1) * band_rows)};
}
std::pair<std::int64_t, std::int64_t> owned(const SegmentInfo& si, const Band& b, int level) {
  const int d = si.out_level - level;
  return {b.o0 >> d, b.o1 >> d};
}
std::pair<std::int64_t, std::int64_t> input_rows(const Geometry& g, const SegmentInfo& si, const Target& t, const Band& bd) {
  const auto [a_t, b_t] = owned(si, bd, t.level);
  const int d = t.level - si.in_level;
  return {std::max<std::int64_t>(0, (a_t >> d) - t.halo_top),
          std::min(g.rows(si.in_level), (b_t >> d) + t.halo_bot)};
}

// Fixed parts of the tiled workspace. The head's output lives at the end of the checkpoint block; its scratch is a
// separate region (the second ping-pong region of the full decode, including the attention's workspace).
struct Fixed {
  std::uint64_t head_x = 0, head_scratch = 0, aux = 0, latents = 0, rgb = 0, cout = 0;
  std::uint64_t total() const { return head_scratch + aux + latents + rgb + cout; }
};
Fixed fixed_parts(int base, std::uint32_t h, std::uint32_t w) {
  const Layout l = layout(base, h, w);
  Fixed f;
  f.head_x = align_up(2ull * h * 2ull * w * static_cast<std::uint64_t>(4 * base) * (vae_fp16_act() ? 2ull : 4ull));
  f.head_scratch = align_up(l.region);
  f.aux = align_up(hip_vae::gn_accum_bytes()) +
          align_up(static_cast<std::uint64_t>(kStatsSlots) * kGroups * 2 * sizeof(float));
  f.latents = align_up(128ull * h * w * 4);
  f.rgb = align_up(16ull * h * 16ull * w * 3);
  f.cout = align_up(16ull * h * 16ull * w * 3 * 4);
  return f;
}

struct SegmentCost {
  double work = 0;
  std::uint64_t block = 0;  // checkpoint block bytes (input and output overlaid)
  std::uint64_t buf = 0;    // bytes of each tile buffer
  std::uint64_t work_bytes = 0;
};
SegmentCost segment_cost(const Geometry& g, const SegmentInfo& si, std::int64_t band_rows) {
  SegmentCost c;
  const std::int64_t n = band_count(g, si, band_rows);
  for (int k = 0; k < si.groupnorms(); ++k) {
    const Target& t = si.targets[static_cast<std::size_t>(k)];
    c.work += read_work(g, t.level, t.ch, g.rows(t.level));  // statistics (inline when n == 1)
    if (n == 1 || t.direct) continue;
    for (std::int64_t j = 0; j < n; ++j) {
      const auto [a, b] = input_rows(g, si, t, band(g, si, band_rows, j));
      c.work += static_cast<double>(b - a) * t.work_per_row + 2.0 * read_work(g, si.in_level, si.in_ch, b - a);
    }
  }
  const Target& out = si.targets.back();
  const std::int64_t ri = static_cast<std::int64_t>(g.row_bytes(si.in_level, si.in_ch));
  const std::int64_t ro = static_cast<std::int64_t>(si.out_ch ? g.row_bytes(si.out_level, si.out_ch) : 0);
  const std::int64_t in_bytes = ri * g.rows(si.in_level), out_bytes = ro * g.rows(si.out_level);
  std::int64_t max_rows = 0, block = std::max(in_bytes, out_bytes);
  std::vector<std::pair<std::int64_t, std::int64_t>> rows(static_cast<std::size_t>(n));
  for (std::int64_t j = 0; j < n; ++j) {
    rows[static_cast<std::size_t>(j)] = input_rows(g, si, out, band(g, si, band_rows, j));
    const std::int64_t r = rows[static_cast<std::size_t>(j)].second - rows[static_cast<std::size_t>(j)].first;
    max_rows = std::max(max_rows, r);
    c.work += static_cast<double>(r) * out.work_per_row + 2.0 * read_work(g, si.in_level, si.in_ch, r);
  }
  if (si.out_ch) c.work += 2.0 * read_work(g, si.out_level, si.out_ch, g.rows(si.out_level));  // commits
  // The input and output checkpoints are overlaid at opposite ends of the block. A write must never reach input rows
  // still needed by a band two steps away, in either sweep direction.
  for (std::int64_t k = 0; k < n; ++k) {
    const Band bk = band(g, si, band_rows, k);
    if (k + 2 < n) block = std::max(block, bk.o1 * ro + in_bytes - rows[static_cast<std::size_t>(k + 2)].first * ri);
    if (k >= 2) block = std::max(block, rows[static_cast<std::size_t>(k - 2)].second * ri + out_bytes - bk.o0 * ro);
  }
  c.block = align_up(static_cast<std::uint64_t>(block));
  c.buf = align_up(static_cast<std::uint64_t>(static_cast<double>(max_rows) * si.buf_per_row));
  c.work_bytes = 3 * c.buf;
  return c;
}

struct PlanSeg {
  int first = 0, last = 0;
  std::int64_t band_rows = 0;
};
struct PlanLayout {
  std::size_t block = 0, work = 0, buf = 0;
  std::vector<const SegmentInfo*> infos;
};

class Planner {
 public:
  Planner(int base, std::uint32_t h, std::uint32_t w)
      : g_{2 * static_cast<std::int64_t>(h), 2 * static_cast<std::int64_t>(w), vae_fp16_act() ? 2u : 4u, kWorkPerByte},
        h_(h), w_(w), base_(base), units_(make_units(base)), fixed_(fixed_parts(base, h, w)) {
    for (int first = 0; first < kTailUnits; ++first)
      for (int last = first + 1; last <= std::min(kTailUnits, first + kMaxSegmentUnits); ++last)
        infos_.push_back(segment_info(g_, units_, first, last));
    for (int u = 0; u < kTailUnits; ++u) {
      std::vector<Op> ops;
      append_ops(units_[static_cast<std::size_t>(u)], ops);
      for (const Op& o : ops) {
        const int cl = op_compute_level(o);
        untiled_work_ += op_work(o, g_) * static_cast<double>(g_.rows(cl) * g_.cols(cl));
        if (o.kind == OpKind::kNorm) untiled_work_ += read_work(g_, o.level, o.in_ch, g_.rows(o.level));
      }
    }
  }

  std::vector<PlanSeg> options(int tile) const {
    std::vector<PlanSeg> opts;
    for (const SegmentInfo& si : infos_) {
      const std::int64_t rows = g_.rows(si.out_level);
      const std::int64_t bh = tile >> (3 - si.out_level);
      if (bh < rows) opts.push_back({si.first, si.last, bh});
      opts.push_back({si.first, si.last, rows});
    }
    return opts;
  }
  static std::vector<const PlanSeg*> best_path(const std::vector<PlanSeg>& opts, std::uint64_t block_cap,
                                               std::uint64_t work_cap, const Geometry& g,
                                               const std::vector<SegmentInfo>& infos) {
    std::array<double, kTailUnits + 1> best;
    std::array<const PlanSeg*, kTailUnits + 1> via{};
    best.fill(std::numeric_limits<double>::infinity());
    best[0] = 0;
    for (const PlanSeg& o : opts) {
      const SegmentInfo& si = info_of(infos, o.first, o.last);
      const SegmentCost cost = segment_cost(g, si, o.band_rows);
      if (cost.block > block_cap || cost.work_bytes > work_cap) continue;
      const double v = best[static_cast<std::size_t>(o.first)] + cost.work;
      if (v < best[static_cast<std::size_t>(o.last)]) {
        best[static_cast<std::size_t>(o.last)] = v;
        via[static_cast<std::size_t>(o.last)] = &o;
      }
    }
    std::vector<const PlanSeg*> path;
    if (!via[kTailUnits]) return path;
    for (int j = kTailUnits; j > 0; j = via[static_cast<std::size_t>(j)]->first) path.push_back(via[static_cast<std::size_t>(j)]);
    std::reverse(path.begin(), path.end());
    return path;
  }
  PlanLayout layout(const std::vector<PlanSeg>& segs, bool validate) const {
    PlanLayout pl;
    int next = 0;
    for (const PlanSeg& s : segs) {
      require(s.first == next && s.last > s.first && s.last <= kTailUnits, "tile plan segments do not chain the tail");
      const SegmentInfo& si = info(s.first, s.last);
      const std::int64_t align = std::int64_t{1} << (si.out_level - si.in_level);
      const std::int64_t rows = g_.rows(si.out_level);
      if (validate && (s.band_rows < 1 || (s.band_rows < rows && s.band_rows % align != 0)))
        throw std::invalid_argument("hip vae: tile plan band height off the band grid");
      const SegmentCost cost = segment_cost(g_, si, s.band_rows);
      pl.block = std::max<std::size_t>(pl.block, static_cast<std::size_t>(cost.block));
      pl.work = std::max<std::size_t>(pl.work, static_cast<std::size_t>(cost.work_bytes));
      pl.buf = std::max<std::size_t>(pl.buf, static_cast<std::size_t>(cost.buf));
      pl.infos.push_back(&si);
      next = s.last;
    }
    require(next == kTailUnits, "tile plan does not cover the tail");
    pl.block = std::max<std::size_t>(pl.block, static_cast<std::size_t>(fixed_.head_x));
    return pl;
  }
  VaeTilePlan finish(int tile, const std::vector<PlanSeg>& segs) const {
    const PlanLayout pl = layout(segs, false);
    VaeTilePlan p;
    p.height = h_;
    p.width = w_;
    p.base_channels = base_;
    p.tile = tile;
    p.fp16_act = vae_fp16_act();
    p.segments.reserve(segs.size());
    double total = 0;
    for (const PlanSeg& s : segs) {
      p.segments.push_back({s.first, s.last, static_cast<std::uint32_t>(s.band_rows)});
      total += segment_cost(g_, info(s.first, s.last), s.band_rows).work;
    }
    p.workspace_bytes = static_cast<std::uint64_t>(pl.block) + pl.work + fixed_.total();
    p.relative_cost = total / untiled_work_;
    return p;
  }
  const Geometry& geometry() const { return g_; }
  const Fixed& fixed() const { return fixed_; }
  const Units& units() const { return units_; }
  const std::vector<SegmentInfo>& infos() const { return infos_; }
  const SegmentInfo& info(int first, int last) const { return info_of(infos_, first, last); }
  static const SegmentInfo& info_of(const std::vector<SegmentInfo>& infos, int first, int last) {
    for (const SegmentInfo& si : infos)
      if (si.first == first && si.last == last) return si;
    throw std::invalid_argument("hip vae: tile plan segment [" + std::to_string(first) + ", " + std::to_string(last) +
                                ") is not a valid segment");
  }

 private:
  Geometry g_;
  std::uint32_t h_, w_;
  int base_;
  Units units_;
  Fixed fixed_;
  std::vector<SegmentInfo> infos_;
  double untiled_work_ = 0;
};

std::vector<std::uint64_t> block_caps(const std::vector<PlanSeg>& opts, const Geometry& g,
                                       const std::vector<SegmentInfo>& infos) {
  std::vector<std::uint64_t> caps;
  for (const PlanSeg& o : opts) caps.push_back(segment_cost(g, Planner::info_of(infos, o.first, o.last), o.band_rows).block);
  std::sort(caps.begin(), caps.end());
  caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
  return caps;
}

void check_tile(int tile) {
  if (tile < 8 || tile % 8 != 0)
    throw std::invalid_argument("hip vae: tile must be a positive multiple of 8 image pixels, got " + std::to_string(tile));
}

}  // namespace

std::string VaeTilePlan::describe() const {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "tile %d px, %.3f GiB, est. %.2fx tail work:", tile,
                static_cast<double>(workspace_bytes) / static_cast<double>(std::uint64_t{1} << 30), relative_cost);
  std::string s = buf;
  const Units units = make_units(base_channels);
  for (const Segment& sg : segments) {
    const int lout = out_level(units[static_cast<std::size_t>(sg.end_unit - 1)]);
    const std::int64_t rows = 2 * static_cast<std::int64_t>(height) << lout;
    s += " [" + std::to_string(sg.first_unit) + "," + std::to_string(sg.end_unit) + ")";
    s += sg.band_rows >= static_cast<std::uint32_t>(rows)
             ? "whole"
             : "x" + std::to_string((rows + sg.band_rows - 1) / sg.band_rows);
  }
  return s;
}


VaeTilePlan VaeDecoder::plan_tiled_min_memory(int base_channels, std::uint32_t h, std::uint32_t w, int tile) {
  require(base_channels > 0 && base_channels % 32 == 0, "decoder widths must be multiples of 32");
  check_tile(tile);
  const Planner p(base_channels, h, w);
  const std::vector<PlanSeg> opts = p.options(tile);
  const std::vector<std::uint64_t> caps = block_caps(opts, p.geometry(), p.infos());
  std::uint64_t best_total = std::numeric_limits<std::uint64_t>::max(), best_block = 0;
  for (const std::uint64_t raw_cap : caps) {
    const std::uint64_t cap = std::max(raw_cap, p.fixed().head_x);  // the head's output lives in the block
    std::array<std::uint64_t, kTailUnits + 1> mm;
    mm.fill(std::numeric_limits<std::uint64_t>::max());
    mm[0] = 0;
    for (const PlanSeg& o : opts) {
      const SegmentCost cost = segment_cost(p.geometry(), p.info(o.first, o.last), o.band_rows);
      if (cost.block <= cap && mm[static_cast<std::size_t>(o.first)] != std::numeric_limits<std::uint64_t>::max())
        mm[static_cast<std::size_t>(o.last)] = std::min(mm[static_cast<std::size_t>(o.last)],
                                                       std::max(mm[static_cast<std::size_t>(o.first)], cost.work_bytes));
    }
    if (mm[kTailUnits] == std::numeric_limits<std::uint64_t>::max()) continue;
    if (cap + mm[kTailUnits] < best_total) {
      best_total = cap + mm[kTailUnits];
      best_block = cap;
    }
  }
  if (best_block == 0) throw std::runtime_error("hip vae: no tiled schedule for this band height");
  const auto path = Planner::best_path(opts, best_block, best_total - best_block, p.geometry(), p.infos());
  if (path.empty()) throw std::runtime_error("hip vae: no tiled schedule for this band height");
  std::vector<PlanSeg> segs;
  for (const PlanSeg* o : path) segs.push_back(*o);
  return p.finish(tile, segs);
}

VaeTilePlan VaeDecoder::plan_tiled_uniform(int base_channels, std::uint32_t h, std::uint32_t w, int tile) {
  require(base_channels > 0 && base_channels % 32 == 0, "decoder widths must be multiples of 32");
  check_tile(tile);
  const Planner p(base_channels, h, w);
  std::vector<PlanSeg> segs;
  for (int first = 0; first < kTailUnits; first += kMaxSegmentUnits) {
    const int last = std::min(kTailUnits, first + kMaxSegmentUnits);
    const SegmentInfo& si = p.info(first, last);
    const std::int64_t rows = p.geometry().rows(si.out_level);
    const std::int64_t bh = tile >> (3 - si.out_level);
    segs.push_back({first, last, bh >= rows ? rows : bh});
  }
  return p.finish(tile, segs);
}


// ---- Exact band-tiled decode: executor ----------------------------------------------------------------------------------

namespace {

template <class Act>
class TiledRunner {
 public:
  static constexpr int kActFmt = ActFmtOf<Act>::value;
  TiledRunner(const VaeDecoder::Weights& wt, Ops& ops, Products products, hipStream_t stream, const Geometry& g,
              const Units& units, char* block, std::size_t block_bytes, std::array<char*, 3> bufs, void* accum,
              float* stats, std::uint8_t* rgb, float* image, float* cout)
      : wt_(wt), ops_(ops), products_(products), stream_(stream), g_(g), units_(units), block_(block),
        block_bytes_(block_bytes), bufs_(bufs), accum_(accum), stats_(stats), rgb_(rgb), image_(image), cout_(cout) {}

  // Runs one segment whose input sits at block offset in_off; returns its output's offset in the block.
  std::size_t run_segment(const SegmentInfo& si, std::int64_t band_rows, std::size_t in_off) {
    const std::int64_t n = band_count(g_, si, band_rows);
    const std::size_t rib = g_.row_bytes(si.in_level, si.in_ch);
    const std::size_t in_bytes = rib * static_cast<std::size_t>(g_.rows(si.in_level));
    const std::size_t out_bytes =
        si.out_ch ? g_.row_bytes(si.out_level, si.out_ch) * static_cast<std::size_t>(g_.rows(si.out_level)) : 0;
    const bool top_down = in_off + in_bytes == block_bytes_;
    require(top_down || in_off == 0, "internal: checkpoint not at either end of the block");
    const std::size_t out_off = top_down ? 0 : block_bytes_ - out_bytes;
    out_ = si.out_ch ? block_ + out_off : nullptr;
    out_level_ = si.out_level;
    char* in = block_ + in_off;

    // One statistics pass per GroupNorm: add its input's owned rows from every band, then finalize with the full
    // element count. The integer accumulators are order-independent, so these equal the whole-tensor statistics.
    if (n > 1) {
      for (int k = 0; k < si.groupnorms(); ++k) {
        const Target& t = si.targets[static_cast<std::size_t>(k)];
        hip_vae::gn_accum_zero(accum_, stream_);
        if (t.direct) {
          hip_vae::gn_accumulate(reinterpret_cast<const float*>(in), g_.rows(si.in_level) * g_.cols(si.in_level), si.in_ch,
                                 accum_, stream_, kActFmt);
        } else {
          for (std::int64_t j = 0; j < n; ++j) {
            const Band bd = band(g_, si, band_rows, j);
            const auto [a, b] = input_rows(g_, si, t, bd);
            Tile tile = start_tile(si, a, b, bd, in_off, 0);
            tile.gn_stop = k;
            if (run_units(si, tile)) throw std::runtime_error("hip vae: statistics pass ran past its GroupNorm");
          }
        }
        finalize(k, t.level, t.ch);
      }
    }

    // Output pass. The next band's input is gathered before the held band's output is committed, so the overlay's
    // block size (checked by the planner) keeps every not-yet-read input row alive.
    int held = -1;
    Tile held_tile;
    for (std::int64_t idx = 0; idx < n; ++idx) {
      const std::int64_t j = top_down ? idx : n - 1 - idx;
      const Band bd = band(g_, si, band_rows, j);
      const auto [a, b] = input_rows(g_, si, si.targets.back(), bd);
      const int gbuf = held < 0 ? 0 : (held + 1) % 3;
      Tile tile = start_tile(si, a, b, bd, in_off, gbuf);
      if (held >= 0) commit(si, held_tile);
      tile.inline_stats = (n == 1);
      if (!run_units(si, tile)) throw std::runtime_error("hip vae: output pass stopped early");
      held = tile.idx[static_cast<std::size_t>(tile.x)];
      held_tile = tile;
    }
    commit(si, held_tile);
    return out_off;
  }

 private:
  struct Tile {
    std::array<int, 3> idx{};  // tile buffers; idx[x] is the current main tensor
    int x = 0, level = 0;
    std::int64_t row0 = 0, rows = 0;  // rows of the tile tensors at `level`
    std::int64_t o0 = 0, o1 = 0;      // owned rows at the segment output level
    int gn_next = 0;
    int gn_stop = -1;           // statistics pass: stop after accumulating this GroupNorm's input
    bool inline_stats = false;  // one band: compute the statistics from the tile itself
  };

  Act* buf(const Tile& t, int k) const {
    return reinterpret_cast<Act*>(bufs_[static_cast<std::size_t>(t.idx[static_cast<std::size_t>((t.x + k) % 3)])]);
  }

  Tile start_tile(const SegmentInfo& si, std::int64_t a, std::int64_t b, const Band& bd, std::size_t in_off, int gbuf) {
    Tile t;
    t.idx = {gbuf, (gbuf + 1) % 3, (gbuf + 2) % 3};
    t.level = si.in_level;
    t.row0 = a;
    t.rows = b - a;
    t.o0 = bd.o0;
    t.o1 = bd.o1;
    const std::size_t rb = g_.row_bytes(si.in_level, si.in_ch);
    copy_async(bufs_[static_cast<std::size_t>(gbuf)], block_ + in_off + static_cast<std::size_t>(a) * rb,
               static_cast<std::size_t>(b - a) * rb, stream_);
    return t;
  }

  float* stats(int k) const { return stats_ + static_cast<std::size_t>(k) * kGroups * 2; }
  void finalize(int k, int level, int ch) {
    if (k >= kStatsSlots) throw std::runtime_error("hip vae: too many GroupNorms in a tile segment");
    const std::int64_t count = g_.rows(level) * g_.cols(level) * (ch / kGroups);
    hip_vae::gn_finalize(accum_, count, kResnetEps, stats(k), stream_);
  }
  void accumulate_owned(const Tile& t, const Act* x, int level, int ch) {
    const int d = out_level_ - level;
    require(d >= 0 && t.row0 >= 0, "internal: GroupNorm below the tile's level");
    const std::int64_t a = t.o0 >> d, b = t.o1 >> d;
    require(a >= t.row0 && b <= t.row0 + t.rows, "internal: tile band outside its buffer");
    const std::size_t off = static_cast<std::size_t>(a - t.row0) * static_cast<std::size_t>(g_.cols(level) * ch);
    hip_vae::gn_accumulate(reinterpret_cast<const float*>(x + off), (b - a) * g_.cols(level), ch, accum_, stream_, kActFmt);
  }
  bool norm(Tile& t, const Act* x, const Norm& n, bool silu, Act* y, int level, int ch) {
    const int k = t.gn_next++;
    if (k == t.gn_stop) {
      accumulate_owned(t, x, level, ch);
      return false;
    }
    if (t.inline_stats) {
      hip_vae::gn_accum_zero(accum_, stream_);
      accumulate_owned(t, x, level, ch);
      finalize(k, level, ch);
    }
    hip_vae::group_norm_apply(reinterpret_cast<const float*>(x), reinterpret_cast<float*>(y), t.rows * g_.cols(level), ch,
                              stats(k), wt_.f32(n.gamma), wt_.f32(n.beta), silu, stream_, kActFmt, kActFmt);
    return true;
  }
  void conv(const Conv& cv, const Act* x, Act* y, int h, int w, const Act* residual = nullptr,
            bool upsample = false) {
    hip_vae::ConvArgs a;
    a.x = reinterpret_cast<const float*>(x);
    a.w = wt_.u16(cv.w);
    a.bias = wt_.f32(cv.b);
    a.residual = reinterpret_cast<const float*>(residual);
    a.out = reinterpret_cast<float*>(y);
    a.x_fmt = kActFmt;
    a.out_fmt = kActFmt;
    a.ldo = cv.k;
    a.h = h;
    a.w_out = w;
    a.c = cv.c;
    a.n = cv.k;
    a.taps = cv.r * cv.r;
    a.upsample = upsample;
    ops_.run("tile conv" + dims(h, w), [&] { hip_vae::conv(a, products_, stream_); });
  }
  // Same op order and buffer roles as Run::resnet() and the CUDA tiled runner.
  bool resnet(Tile& t, const Resnet& r) {
    const int rows = static_cast<int>(t.rows), cols = static_cast<int>(g_.cols(t.level));
    Act* xt = buf(t, 0);
    Act* bt = buf(t, 2);
    if (r.shortcut.k != 0) {
      if (t.gn_next == t.gn_stop) return norm(t, xt, r.norm1, true, xt, t.level, r.in);
      Act* st = buf(t, 1);
      conv(r.shortcut, xt, st, rows, cols);
      norm(t, xt, r.norm1, true, xt, t.level, r.in);
      conv(r.conv1, xt, bt, rows, cols);
      if (!norm(t, bt, r.norm2, true, bt, t.level, r.out)) return false;
      conv(r.conv2, bt, st, rows, cols, st);
      t.x = (t.x + 1) % 3;
      return true;
    }
    Act* at = buf(t, 1);
    if (!norm(t, xt, r.norm1, true, at, t.level, r.in)) return false;
    conv(r.conv1, at, bt, rows, cols);
    if (!norm(t, bt, r.norm2, true, bt, t.level, r.out)) return false;
    conv(r.conv2, bt, xt, rows, cols, xt);
    return true;
  }
  bool unit(Tile& t, int u) {
    const UnitShape& us = units_[static_cast<std::size_t>(u)];
    switch (us.kind) {
      case UnitKind::kResnet:
        return resnet(t, u == 0 ? wt_.mid[1] : wt_.up[(u - 1) / 4][(u - 1) % 4]);
      case UnitKind::kUpsample: {
        const Act* xt = buf(t, 0);
        ++t.level;
        t.row0 *= 2;
        t.rows *= 2;
        Act* out = buf(t, 1);  // the 2x upsample is folded into conv's im2col, so no upsampled buffer is needed
        conv(wt_.upsampler[(u - 1) / 4], xt, out, static_cast<int>(t.rows), static_cast<int>(g_.cols(t.level)), nullptr,
             true);
        t.x = (t.x + 1) % 3;
        return true;
      }
      case UnitKind::kOut: {
        Act* xt = buf(t, 0);
        if (!norm(t, xt, wt_.norm_out, true, xt, t.level, wt_.norm_out.c)) return false;
        const int cols = static_cast<int>(g_.cols(t.level));
        float* o = cout_ + static_cast<std::size_t>(t.row0) * static_cast<std::size_t>(cols) * 3;
        ops_.run("tile conv_out" + dims(static_cast<int>(t.rows), cols), [&] {
          hip_vae::conv_narrow(reinterpret_cast<const float*>(xt), wt_.f32(wt_.conv_out.w), wt_.f32(wt_.conv_out.b), o,
                               static_cast<int>(t.rows), cols, wt_.conv_out.c, 3, stream_,
                               round_of(products_), kActFmt);
        });
        t.x = (t.x + 1) % 3;
        return true;
      }
    }
    return false;
  }
  bool run_units(const SegmentInfo& si, Tile& t) {
    for (int u = si.first; u < si.last; ++u)
      if (!unit(t, u)) return false;
    return true;
  }
  // Writes only the band's owned rows to the checkpoint or the RGB8 buffer.
  void commit(const SegmentInfo& si, const Tile& t) {
    require(t.level == si.out_level, "internal: band is not at the segment's output level");
    const std::int64_t a = t.o0, b = t.o1;
    const int cols = static_cast<int>(g_.cols(t.level));
    const std::size_t off = static_cast<std::size_t>(a - t.row0);
    require(a >= t.row0 && b <= t.row0 + t.rows, "internal: commit outside its buffer");
    if (si.out_ch == 0) {
      const float* src = cout_ + static_cast<std::size_t>(a) * static_cast<std::size_t>(cols) * 3;
      const std::size_t n = static_cast<std::size_t>(b - a) * static_cast<std::size_t>(cols) * 3;
      ops_.run("tile rgb8", [&] { hip_vae::rgb8(src, rgb_ + static_cast<std::size_t>(a) * static_cast<std::size_t>(cols) * 3,
                                                static_cast<std::int64_t>(n), stream_); });
      if (image_ != nullptr)
        copy_async(image_ + static_cast<std::size_t>(a) * static_cast<std::size_t>(cols) * 3, src, n * 4, stream_);
      return;
    }
    const float* src = reinterpret_cast<const float*>(buf(t, 0) + off * static_cast<std::size_t>(cols * si.out_ch));
    const std::size_t rb = g_.row_bytes(si.out_level, si.out_ch);
    copy_async(out_ + static_cast<std::size_t>(a) * rb, src, static_cast<std::size_t>(b - a) * rb, stream_);
  }

  const VaeDecoder::Weights& wt_;
  Ops& ops_;
  Products products_;
  hipStream_t stream_;
  const Geometry& g_;
  const Units& units_;
  char* block_;
  std::size_t block_bytes_;
  std::array<char*, 3> bufs_;
  void* accum_;
  float* stats_;
  std::uint8_t* rgb_;
  float* image_;
  float* cout_;
  char* out_ = nullptr;
  int out_level_ = 0;
};

}  // namespace

template <class Act>
VaeDecodeStats tiled_impl(const VaeDecoder::Weights& wt, Products products, bool profile, int c, const float* latents,
                          std::uint32_t h, std::uint32_t w, std::vector<std::uint8_t>& rgb, const VaeTilePlan& plan,
                          std::vector<float>* image, const Layout& l, std::uint64_t ws_bytes) {
  const int mid = 4 * c;
  const Planner p(c, h, w);
  std::vector<PlanSeg> segs;
  segs.reserve(plan.segments.size());
  for (const VaeTilePlan::Segment& sg : plan.segments) segs.push_back({sg.first_unit, sg.end_unit, sg.band_rows});
  const PlanLayout pl = p.layout(segs, true);
  const Fixed& f = p.fixed();
  const int hh = static_cast<int>(2 * h), ww = static_cast<int>(2 * w), H = static_cast<int>(16 * h),
            W = static_cast<int>(16 * w);
  const std::size_t image_bytes = image != nullptr ? static_cast<std::size_t>(H) * W * 3 * 4 : 0;
  Stream stream;
  DeviceBuffer ws(static_cast<std::size_t>(ws_bytes));
  DeviceBuffer image_stage(image_bytes);
  struct Drain {
    hipStream_t s;
    ~Drain() { (void)hipStreamSynchronize(s); }
  } drain{stream.get()};
  char* block = static_cast<char*>(ws.data());
  char* work = block + pl.block;
  char* head_scratch = work + pl.work;
  char* aux = head_scratch + f.head_scratch;
  float* z_dev = reinterpret_cast<float*>(aux + f.aux);
  std::uint8_t* rgb_stage = reinterpret_cast<std::uint8_t*>(z_dev + 128ull * h * w);
  float* cout_dev = reinterpret_cast<float*>(reinterpret_cast<char*>(rgb_stage) + f.rgb);
  copy_async(z_dev, latents, 128ull * h * w * 4, stream.get());
  void* accum = aux;
  float* stats = reinterpret_cast<float*>(aux + align_up(hip_vae::gn_accum_bytes()));
  const std::size_t in_off = static_cast<std::size_t>(pl.block - f.head_x);

  Ops ops(stream.get(), profile);
  run_head_ops<Act>(wt, ops, products, stream.get(), z_dev, l, reinterpret_cast<Act*>(block + in_off), head_scratch,
                    accum, stats, mid, hh, ww, h, w);

  const Geometry g{2 * static_cast<std::int64_t>(h), 2 * static_cast<std::int64_t>(w), vae_fp16_act() ? 2u : 4u,
                   kWorkPerByte};
  const std::uint64_t buf = pl.buf;
  TiledRunner<Act> runner(wt, ops, products, stream.get(), g, p.units(), block, pl.block,
                          {work, work + buf, work + 2 * buf}, accum, stats, rgb_stage,
                          image != nullptr ? image_stage.as<float>() : nullptr, cout_dev);
  std::size_t off = in_off;
  for (std::size_t i = 0; i < segs.size(); ++i) off = runner.run_segment(*pl.infos[i], segs[i].band_rows, off);

  stream.synchronize();
  const std::size_t n = static_cast<std::size_t>(H) * W * 3;
  rgb.resize(n);
  ws.download(rgb.data(), n, reinterpret_cast<char*>(rgb_stage) - block);
  if (image != nullptr) {
    image->resize(n);
    image_stage.download(image->data(), n * 4);
  }
  VaeDecodeStats st;
  if (profile) st.profile = ops.profile();
  return st;
}

VaeDecodeStats VaeDecoder::decode_tiled_exact(const float* latents, std::uint32_t h, std::uint32_t w,
                                               std::vector<std::uint8_t>& rgb, const VaeTilePlan& plan,
                                               std::vector<float>* image) {
  const auto t0 = std::chrono::steady_clock::now();
  const int c = base_;
  device_.make_current();
  vae_round_flag() = round_of(conv_products_);
  vae_act_fmt_flag() = act_fmt_;
  const Layout l = layout(c, h, w);  // validates the latent grid
  require(plan.height == h && plan.width == w && plan.base_channels == c,
          "tile plan was made for another latent grid or decoder width");
  const Planner p(c, h, w);
  std::vector<PlanSeg> segs;
  segs.reserve(plan.segments.size());
  for (const VaeTilePlan::Segment& s : plan.segments) segs.push_back({s.first_unit, s.end_unit, s.band_rows});
  const PlanLayout pl = p.layout(segs, true);
  const Fixed& f = p.fixed();
  // A plan made before the decoder was constructed was stamped with the default (fp32) activation mode. Its estimate
  // is still a safe upper bound, but the workspace must be sized for the decoder's actual mode. Only compare against
  // the estimate when both modes agree; otherwise recompute below.
  const std::uint64_t ws_bytes = static_cast<std::uint64_t>(pl.block) + pl.work + f.total();
  if (plan.fp16_act == vae_fp16_act())
    require(ws_bytes == plan.workspace_bytes, "tile plan workspace out of date");
  const std::size_t H = 16ull * h, W = 16ull * w;
  const std::size_t image_bytes = image != nullptr ? H * W * 3 * 4 : 0;
  const std::uint64_t need = ws_bytes + image_bytes;  // the fp32 image needs its own staging buffer
  const std::uint64_t free = device_.free_memory();
  if (need > free) {
    char msg[256];
    std::snprintf(msg, sizeof msg, "the band-tiled VAE decode needs %.2f GiB of device memory, %s has %.2f GiB free",
                  static_cast<double>(need) / (1u << 30), device_.info().name.c_str(),
                  static_cast<double>(free) / (1u << 30));
    throw std::runtime_error(msg);
  }
  VaeDecodeStats st;
  if (act_fmt_ == 2)
    st = tiled_impl<bf16_bits>(*weights_, conv_products_, profile_, c, latents, h, w, rgb, plan, image, l, ws_bytes);
  else if (act_fmt_ == 1)
    st = tiled_impl<fp16_bits>(*weights_, conv_products_, profile_, c, latents, h, w, rgb, plan, image, l, ws_bytes);
  else
    st = tiled_impl<float>(*weights_, conv_products_, profile_, c, latents, h, w, rgb, plan, image, l, ws_bytes);
  st.device_bytes = need;
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return st;
}

}  // namespace mi::hip
