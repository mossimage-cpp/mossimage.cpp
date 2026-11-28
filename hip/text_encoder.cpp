// The standard headers first: HIP's define __noinline__ as a macro, which breaks libstdc++'s <chrono> (GCC 16) included
// after them.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "text_encoder.h"

#include "bf16_gemm.h"
#include "dequant_fast.h"
#include "half.h"
#include "hip_block.h"
#include "safetensors.h"
#include "te_dequant.h"
#include "te_kernels.h"
#include "weights/quant/quant_codec.h"

namespace mi::hip {
namespace {

namespace te = hip_block::te;
using Clock = std::chrono::steady_clock;
using bf16 = std::uint16_t;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("hip text encoder: " + what); }

constexpr int kQgDim = 2 * kTeHeads * kTeHeadDim;                           // q_proj: per head [q(256) | gate(256)]
constexpr int kKvDim = kTeKvHeads * kTeHeadDim;                             // 1024
constexpr int kQkvDim = kQgDim + 2 * kKvDim;                                // 10240
constexpr int kBaDim = 2 * te::kGdnValueHeads;                              // 64
constexpr int kQkvzbaDim = te::kGdnConvDim + te::kGdnValueDim + kBaDim;      // [qkv | z | b | a], 12352
constexpr int kGateUpDim = 2 * kTeIntermediate;                             // 24576
constexpr std::uint64_t kReadChunk = std::uint64_t{16} << 20;

std::uint64_t align256(std::uint64_t v) { return (v + 255) & ~std::uint64_t{255}; }

using TensorRef = std::pair<const st::File*, const st::TensorInfo*>;

// A weight's source: a file tensor (BF16), a dequantized I8 weight held as host BF16, or a raw int8 weight whose
// dequantization will run on the device (host_bf16 then points at int8 bytes and `bytes` is the int8 byte count).
struct WeightSource {
  const st::File* file = nullptr;      // set for file tensors
  const st::TensorInfo* info = nullptr;  // set for file tensors
  const bf16* host_bf16 = nullptr;     // set for dequantized I8 weights or device-dequant int8 weights
  std::uint64_t bytes = 0;
  bool is_i8 = false;                  // raw int8, to be dequantized on the device
  std::uint64_t scale_off = 0;         // into the device scale array
  std::uint64_t footprint() const { return is_i8 ? 2 * bytes : bytes; }
};

// One read of a tensor into the layer's staging layout.
struct Piece {
  WeightSource src;
  std::uint64_t offset;  // in the layer layout
};

// A projection group: n rows of k, its tensors' rows adjacent in the layer layout at `raw`; `packed` its place in the
// packed buffer (pack_rows(n) rows).
struct Gemm {
  std::uint64_t raw = 0, packed = 0;
  int n = 0, k = 0;
};

// Where a layer's tensors sit. The small tensors (norms, conv, gates' parameters) come first, at the same offsets in the
// raw and the packed layout; the projection groups follow.
struct I8Piece {
  std::uint64_t offset, scale_off;  // into the raw layer buffer / the device scale array
  int n, k;
};
struct CopyRange {
  std::uint64_t offset, bytes;  // a bf16 piece copied verbatim from the int8 staging buffer
};

struct LayerPlan {
  bool full_attention = false;
  std::vector<Piece> pieces;
  std::vector<I8Piece> i8_pieces;    // device-dequantized raw int8 weights in this layer
  std::vector<CopyRange> d2d_ranges; // bf16 pieces copied from the int8 staging buffer on the device
  std::uint64_t bytes = 0, small_bytes = 0, packed_bytes = 0;
  std::uint64_t in_norm = 0, post_norm = 0;
  std::uint64_t conv = 0, a_log = 0, dt_bias = 0, gdn_norm = 0;  // GatedDeltaNet
  std::uint64_t q_norm = 0, k_norm = 0;                          // attention
  Gemm qkvzba, out, qkv, o, gate_up, down;
};

// Activations for `t` tokens, offsets in one device buffer, laid out per phase of a layer.
struct Scratch {
  std::uint64_t x, h, pack, phase, end;
  // GatedDeltaNet
  std::uint64_t qkvzba, conv, g, beta, core, ws;
  // attention
  std::uint64_t qkv, q, attn;
  // MLP
  std::uint64_t gu, act;
  Scratch(std::uint32_t t, bool packed) {
    const std::uint64_t T = t;
    std::uint64_t at = 0;
    const auto take = [&](std::uint64_t bytes) {
      const std::uint64_t o = at;
      at = align256(at + bytes);
      return o;
    };
    x = take(T * kTeHidden * 2);
    h = take(T * kTeHidden * 2);
    pack = take(packed ? static_cast<std::uint64_t>(te::pack_rows(t)) * kTeIntermediate * 2 : 0);
    phase = at;
    qkvzba = take(T * kQkvzbaDim * 2);
    conv = take(T * te::kGdnConvDim * 2);
    g = take(T * te::kGdnValueHeads * 4);
    beta = take(T * te::kGdnValueHeads * 2);
    core = take(T * te::kGdnValueDim * 2);
    ws = take(te::GdnWorkspace::bytes(t));
    const std::uint64_t gdn_end = at;
    at = phase;
    qkv = take(T * kQkvDim * 2);
    q = take(T * kTeHidden * 2);
    attn = take(T * kTeHidden * 2);
    const std::uint64_t attn_end = at;
    at = phase;
    gu = take(T * kGateUpDim * 2);
    act = take(T * kTeIntermediate * 2);
    end = std::max({gdn_end, attn_end, at});
  }
};

// The reader thread's handoff: staging slot s holds layer filled[s] once read; the main thread frees it after its copy.
struct Handoff {
  std::mutex mu;
  std::condition_variable cv;
  int filled[2] = {-1, -1};
  bool busy[2] = {false, false};
  bool abort = false;
  std::exception_ptr error;
};

}  // namespace

struct TextEncoder::Impl {
  Device& device;
  bool current;  // the device made current before any member below allocates on it (the streams first)
  TextEncoderOptions options;
  std::map<std::string, std::unique_ptr<st::File>> files;
  std::map<std::string, WeightSource> tensors;
  WeightSource embed{}, final_norm{};
  HostBuffer dequant;    // dequantized I8 weights that stay on the host (embedding, final norm), as BF16
  HostBuffer i8_host;    // raw int8 weights for device-side dequantization
  HostBuffer scales_host;
  DeviceBuffer scales_dev;
  DeviceBuffer i8dev;  // H2D staging for the int8 layer weights; one buffer, ordered by dequant_free
  std::int64_t vocab = 0;
  std::vector<LayerPlan> layers;
  std::uint64_t layer_bytes = 0, packed_bytes = 0;
  Stream stream, copy_stream;
  HostBuffer staging[2];
  DeviceBuffer raw[2], packed;  // raw[1] only on the row-major path
  DeviceBuffer consts;          // the final norm weight, then bf16 ones [4096]
  TextEncoderStats stats;

  Impl(Device& d, const std::filesystem::path& dir, TextEncoderOptions o)
      : device(d), current((d.make_current(), true)), options(o) {
    if (options.reader_threads < 1) throw std::invalid_argument("hip text encoder: reader_threads must be positive");
    // `dir` may be a bundle's text_encoder/ directory or a single .safetensors file (--te).
    for (const auto& [name, shard] : st::index_files(dir)) {
      // Accept both "language_model.*" and "model.language_model.*" tensor naming; the file lookup keeps the original name.
      std::string key = name;
      if (key.rfind("model.", 0) == 0) key = key.substr(6);
      if (key.rfind("language_model.", 0) != 0) continue;
      const std::u8string shard_u8 = shard.u8string();
      const std::string shard_key(shard_u8.begin(), shard_u8.end());
      auto& file = files[shard_key];
      if (!file) file = std::make_unique<st::File>(shard);
      tensors[key] = {file.get(), &file->at(name), nullptr, file->at(name).bytes};
    }
    // The high tier's text encoder ships int8 ConvRot weights. By default the layer weights stay int8 on the device and
    // te_dequant.hip runs the radix-4 transform there; the embedding and final norm stay on the host path because they
    // are read outside a layer. Tensors are independent, so the file reads run in parallel.
    const bool devdq = options.device_dequant && !options.exact_dequant;
    std::vector<std::string> host_keys, dev_keys;
    std::uint64_t host_i8_bytes = 0, dev_i8_bytes = 0, scale_elems = 0;
    for (const auto& [key, src] : tensors) {
      if (!src.info || src.info->dtype != "I8") continue;
      const bool host_only = key == "language_model.embed_tokens.weight" || key == "language_model.norm.weight";
      if (devdq && !host_only) {
        dev_keys.push_back(key);
        dev_i8_bytes += src.info->bytes;
        scale_elems += static_cast<std::uint64_t>(src.info->shape[0]);
      } else {
        host_keys.push_back(key);
        host_i8_bytes += src.info->bytes;
      }
    }
    dequant = HostBuffer(static_cast<std::size_t>(2 * host_i8_bytes));
    i8_host = HostBuffer(static_cast<std::size_t>(dev_i8_bytes));
    scales_host = HostBuffer(static_cast<std::size_t>(4 * scale_elems));
    bf16* dq = dequant.as<bf16>();
    char* i8p = static_cast<char*>(i8_host.data());
    float* sch = scales_host.as<float>();
    std::vector<std::size_t> host_off(host_keys.size()), dev_off(dev_keys.size()), scale_off(dev_keys.size());
    std::size_t dq_off = 0, db_off = 0, sc_off = 0;
    for (std::size_t i = 0; i < host_keys.size(); ++i) {
      host_off[i] = dq_off;
      dq_off += static_cast<std::size_t>(tensors[host_keys[i]].info->bytes);
    }
    for (std::size_t i = 0; i < dev_keys.size(); ++i) {
      dev_off[i] = db_off;
      db_off += static_cast<std::size_t>(tensors[dev_keys[i]].info->bytes);
      scale_off[i] = sc_off;
      sc_off += static_cast<std::size_t>(tensors[dev_keys[i]].info->shape[0]);
    }
    // hardware_concurrency() is allowed to return 0; at least the calling thread must be present so that
    // errors[] has room for the exception work(0) stores there.
    const int threads = std::max(1, std::min<int>(8, static_cast<int>(std::thread::hardware_concurrency())));
    std::atomic<std::size_t> next{0};
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(threads));
    const auto work = [&](int t) {
      try {
        for (std::size_t i = next++; i < host_keys.size() + dev_keys.size(); i = next++) {
          const bool dev = i >= host_keys.size();
          const std::size_t j = dev ? i - host_keys.size() : i;
          const std::string& key = dev ? dev_keys[j] : host_keys[j];
          const WeightSource& src = tensors[key];
          const std::int64_t N = src.info->shape[0], K = src.info->shape[1];
          const auto sit = tensors.find(key + "_scale");
          if (sit == tensors.end() || !sit->second.file) fail("no weight_scale for " + key);
          std::vector<float> scale(static_cast<std::size_t>(N));
          sit->second.file->read(*sit->second.info, scale.data());
          if (dev) {
            src.file->read(*src.info, i8p + dev_off[j]);  // straight into the persistent int8 host image
            std::memcpy(sch + scale_off[j], scale.data(), static_cast<std::size_t>(N) * sizeof(float));
          } else {
            std::vector<std::int8_t> q(static_cast<std::size_t>(N) * K);
            src.file->read(*src.info, q.data());
            if (options.exact_dequant) {
              std::vector<float> f32(static_cast<std::size_t>(N) * K);
              mi::quant::dequantize_rows(q.data(), scale.data(), N, K, 256, f32.data());
              for (std::size_t k = 0; k < f32.size(); ++k) dq[host_off[j] + k] = mi::quant::float_to_bf16(f32[k]);
            } else {
              mi::hip::dequantize_rows_fast_bf16(q.data(), scale.data(), N, K, 256, dq + host_off[j], 1);
            }
          }
        }
      } catch (...) {
        errors[static_cast<std::size_t>(t)] = std::current_exception();
        next = host_keys.size() + dev_keys.size();
      }
    };
    {
      std::vector<std::jthread> helpers;
      for (int t = 1; t < threads; ++t) helpers.emplace_back(work, t);
      work(0);
    }
    for (const auto& e : errors) {
      if (e) std::rethrow_exception(e);
    }
    for (std::size_t i = 0; i < host_keys.size(); ++i) {
      const std::string& key = host_keys[i];
      const std::size_t count = static_cast<std::size_t>(tensors[key].info->bytes);
      tensors[key] = {nullptr, nullptr, dq + host_off[i], 2 * count};
    }
    for (std::size_t i = 0; i < dev_keys.size(); ++i) {
      const std::string& key = dev_keys[i];
      const std::size_t count = static_cast<std::size_t>(tensors[key].info->bytes);
      tensors[key] = {nullptr, nullptr, reinterpret_cast<const bf16*>(i8p + dev_off[i]), count, true, scale_off[i]};
    }
    if (devdq && scale_elems != 0) {
      scales_dev = DeviceBuffer(static_cast<std::size_t>(4 * scale_elems));
      scales_dev.upload(sch, static_cast<std::size_t>(4 * scale_elems));
    }
    const auto it = tensors.find("language_model.embed_tokens.weight");
    if (it == tensors.end() || it->second.bytes == 0 || it->second.bytes % (2 * kTeHidden) != 0) {
      fail("language_model.embed_tokens.weight must be [vocab, 4096]");
    }
    embed = it->second;
    vocab = static_cast<std::int64_t>(embed.bytes / (2 * kTeHidden));
    final_norm = lookup("language_model.norm.weight", {kTeHidden});
    for (int i = 0; i < kTeLayers; ++i) layers.push_back(plan_layer(i));
    for (const LayerPlan& l : layers) {
      layer_bytes = std::max(layer_bytes, l.bytes);
      packed_bytes = std::max(packed_bytes, l.packed_bytes);
    }
    staging[0] = HostBuffer(layer_bytes);
    staging[1] = HostBuffer(layer_bytes);
    raw[0] = DeviceBuffer(layer_bytes);
    if (options.packed && !options.tiled_gemm) {
      packed = DeviceBuffer(packed_bytes);
    } else {
      raw[1] = DeviceBuffer(layer_bytes);
    }
    if (devdq) i8dev = DeviceBuffer(layer_bytes);
    std::vector<bf16> c(2 * kTeHidden, 0x3f80);  // [final norm | ones]
    if (final_norm.file) {
      final_norm.file->read(*final_norm.info, c.data());
    } else {
      std::memcpy(c.data(), final_norm.host_bf16, final_norm.bytes);
    }
    consts = DeviceBuffer(c.size() * 2);
    consts.upload(c.data(), c.size() * 2);
    device.synchronize();  // a pageable upload's DMA may be in flight on return
  }

  const WeightSource& lookup(const std::string& name, const std::vector<std::int64_t>& shape) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) fail("no tensor " + name);
    const WeightSource& s = it->second;
    std::uint64_t expected = s.is_i8 ? 1 : 2;  // int8: one byte per element; BF16: two
    for (const std::int64_t d : shape) expected *= static_cast<std::uint64_t>(d);
    if (s.bytes != expected) fail("tensor " + name + " has the wrong size");
    return s;
  }

  LayerPlan plan_layer(int i) const {
    LayerPlan l;
    l.full_attention = i % 4 == 3;
    const std::string p = "language_model.layers." + std::to_string(i) + ".";
    const auto add = [&](const std::string& name, const std::vector<std::int64_t>& shape) {
      const WeightSource& t = lookup(p + name, shape);
      l.bytes = align256(l.bytes);
      const std::uint64_t at = l.bytes;
      l.pieces.push_back({t, at});
      l.bytes += t.footprint();
      if (t.is_i8) {
        if (shape.size() != 2) fail("device-dequant int8 weight must be 2D: " + name);
        l.i8_pieces.push_back({at, t.scale_off, static_cast<int>(shape[0]), static_cast<int>(shape[1])});
      } else {
        l.d2d_ranges.push_back({at, t.footprint()});
      }
      return at;
    };
    // A group's tensors back to back: their sizes are multiples of 256 bytes, so the alignment adds no gap.
    const auto group = [&](const std::vector<std::pair<std::string, int>>& parts, int k) {
      Gemm g;
      g.k = k;
      g.raw = align256(l.bytes);
      for (const auto& [name, rows] : parts) {
        const std::uint64_t at = add(name, {rows, k});
        if (at != g.raw + static_cast<std::uint64_t>(g.n) * static_cast<std::uint64_t>(k) * 2) fail("internal: group layout");
        g.n += rows;
      }
      return g;
    };
    l.in_norm = add("input_layernorm.weight", {kTeHidden});
    l.post_norm = add("post_attention_layernorm.weight", {kTeHidden});
    if (l.full_attention) {
      l.q_norm = add("self_attn.q_norm.weight", {kTeHeadDim});
      l.k_norm = add("self_attn.k_norm.weight", {kTeHeadDim});
      l.small_bytes = align256(l.bytes);
      l.qkv = group({{"self_attn.q_proj.weight", kQgDim}, {"self_attn.k_proj.weight", kKvDim}, {"self_attn.v_proj.weight", kKvDim}},
                    kTeHidden);
      l.o = group({{"self_attn.o_proj.weight", kTeHidden}}, kTeHeads * kTeHeadDim);
    } else {
      l.conv = add("linear_attn.conv1d.weight", {te::kGdnConvDim, 1, te::kGdnConvKernel});
      l.a_log = add("linear_attn.A_log", {te::kGdnValueHeads});
      l.dt_bias = add("linear_attn.dt_bias", {te::kGdnValueHeads});
      l.gdn_norm = add("linear_attn.norm.weight", {te::kGdnHeadDim});
      l.small_bytes = align256(l.bytes);
      l.qkvzba = group({{"linear_attn.in_proj_qkv.weight", te::kGdnConvDim},
                        {"linear_attn.in_proj_z.weight", te::kGdnValueDim},
                        {"linear_attn.in_proj_b.weight", te::kGdnValueHeads},
                        {"linear_attn.in_proj_a.weight", te::kGdnValueHeads}},
                       kTeHidden);
      l.out = group({{"linear_attn.out_proj.weight", kTeHidden}}, te::kGdnValueDim);
    }
    l.gate_up = group({{"mlp.gate_proj.weight", kTeIntermediate}, {"mlp.up_proj.weight", kTeIntermediate}}, kTeHidden);
    l.down = group({{"mlp.down_proj.weight", kTeHidden}}, kTeIntermediate);
    l.bytes = align256(l.bytes);
    // The packed layout: the small tensors, then each group's pack_rows(n) rows.
    l.packed_bytes = l.small_bytes;
    for (Gemm* g : {&l.qkvzba, &l.out, &l.qkv, &l.o, &l.gate_up, &l.down}) {
      if (g->n == 0) continue;
      g->packed = l.packed_bytes;
      l.packed_bytes = align256(l.packed_bytes + static_cast<std::uint64_t>(te::pack_rows(g->n)) * static_cast<std::uint64_t>(g->k) * 2);
    }
    return l;
  }

  // A layer's tensors into `dst` at their offsets, in chunks of 16 MiB taken in turn by `reader_threads` threads.
  void read_layer(const LayerPlan& l, std::byte* dst) const {
    struct Chunk {
      const Piece* piece;
      std::uint64_t rel, bytes;
    };
    std::vector<Chunk> chunks;
    for (const Piece& p : l.pieces) {
      for (std::uint64_t rel = 0; rel < p.src.bytes; rel += kReadChunk) {
        chunks.push_back({&p, rel, std::min(kReadChunk, p.src.bytes - rel)});
      }
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(options.reader_threads));
    const auto work = [&](std::size_t t) {
      try {
        for (std::size_t i = next++; i < chunks.size(); i = next++) {
          const Chunk& c = chunks[i];
          const WeightSource& s = c.piece->src;
          if (s.file) {
            s.file->read(*s.info, dst + c.piece->offset + c.rel, c.rel, c.bytes);
          } else if (s.is_i8) {
            std::memcpy(dst + c.piece->offset + c.rel, reinterpret_cast<const char*>(s.host_bf16) + c.rel, c.bytes);
          } else {
            std::memcpy(dst + c.piece->offset + c.rel, s.host_bf16 + c.rel / 2, c.bytes);
          }
        }
      } catch (...) {
        errors[t] = std::current_exception();
        next = chunks.size();
      }
    };
    {
      std::vector<std::jthread> helpers;
      for (std::size_t t = 1; t < errors.size(); ++t) helpers.emplace_back(work, t);
      work(0);
    }
    for (const auto& e : errors) {
      if (e) std::rethrow_exception(e);
    }
  }

  std::vector<std::vector<bf16>> encode(const std::vector<std::vector<std::int32_t>>& seqs, const TeLayerHook& hook);
};

TextEncoder::TextEncoder(Device& device, const std::filesystem::path& dir, TextEncoderOptions options)
    : impl_(std::make_unique<Impl>(device, dir, options)) {}

TextEncoder::~TextEncoder() = default;

std::int64_t TextEncoder::vocab() const { return impl_->vocab; }
const TextEncoderStats& TextEncoder::last_stats() const { return impl_->stats; }

std::vector<std::vector<std::uint16_t>> TextEncoder::encode(const std::vector<std::vector<std::int32_t>>& sequences,
                                                            const TeLayerHook& hook) {
  return impl_->encode(sequences, hook);
}

std::vector<std::vector<bf16>> TextEncoder::Impl::encode(const std::vector<std::vector<std::int32_t>>& seqs,
                                                         const TeLayerHook& hook) {
  const auto t_start = Clock::now();
  device.make_current();
  if (seqs.empty()) throw std::invalid_argument("hip text encoder: no sequences");
  std::vector<std::int32_t> cu = {0};
  std::uint32_t max_len = 0;
  for (const auto& s : seqs) {
    if (s.empty()) throw std::invalid_argument("hip text encoder: empty sequence");
    for (const std::int32_t id : s) {
      if (id < 0 || id >= vocab) {
        throw std::invalid_argument("hip text encoder: token id " + std::to_string(id) + " outside the vocabulary");
      }
    }
    if (static_cast<std::uint64_t>(cu.back()) + s.size() > kTeMaxTokens) {
      throw std::invalid_argument("hip text encoder: more than 65535 tokens");
    }
    cu.push_back(cu.back() + static_cast<std::int32_t>(s.size()));
    max_len = std::max(max_len, static_cast<std::uint32_t>(s.size()));
  }
  const auto T = static_cast<std::uint32_t>(cu.back());
  const int nseq = static_cast<int>(seqs.size());
  const bool packed_path = options.packed && !options.tiled_gemm;
  const bool devdq = options.device_dequant && !options.exact_dequant;
  const int nraw = packed_path ? 1 : 2;
  stats = TextEncoderStats{};
  stats.tokens = T;

  const Scratch sc(T, packed_path);
  DeviceBuffer act(sc.end), cu_dev(cu.size() * 4), rope(static_cast<std::size_t>(max_len) * te::kRopeDim * 2);
  stats.device_bytes = static_cast<std::uint64_t>(nraw + (devdq ? 1 : 0)) * layer_bytes +
                       (packed_path ? packed_bytes : 0) + sc.end + cu_dev.bytes() + rope.bytes() + consts.bytes() +
                       scales_dev.bytes();
  stats.pinned_bytes = 2 * layer_bytes;
  cu_dev.upload(cu.data(), cu.size() * 4);
  auto* base = act.as<std::byte>();
  const auto at = [&](std::uint64_t offset) { return reinterpret_cast<bf16*>(base + offset); };
  bf16 *const x = at(sc.x), *const h = at(sc.h), *const xp = at(sc.pack);
  const bf16* final_norm_w = consts.as<bf16>();
  const bf16* ones = consts.as<bf16>() + kTeHidden;
  const hipStream_t s = stream.get(), cs = copy_stream.get();
  const auto* cu_d = cu_dev.as<std::int32_t>();
  const auto* rope_d = rope.as<bf16>();
  te::rope_table(rope.as<bf16>(), max_len, s);  // positions 0 .. max_len - 1, as the engine's qwen_rope_table

  // Per-op device times (TextEncoderOptions::time_ops): events before and after each op.
  struct OpTiming {
    const char* name;
    std::unique_ptr<Event> start, end;
  };
  std::vector<OpTiming> op_events;
  const auto op = [&](const char* name, auto&& run) {
    if (!options.time_ops) {
      run();
      return;
    }
    OpTiming e{name, std::make_unique<Event>(true), std::make_unique<Event>(true)};
    e.start->record(s);
    run();
    e.end->record(s);
    op_events.push_back(std::move(e));
  };
  const auto for_each_sequence = [&](auto&& f) {
    for (int i = 0; i < nseq; ++i) f(cu[static_cast<std::size_t>(i)], cu[static_cast<std::size_t>(i) + 1] - cu[static_cast<std::size_t>(i)]);
  };
  const auto hidden_norm = [&](const bf16* weight, bf16* out) {
    op("rms_norm", [&] {
      for_each_sequence([&](std::int32_t s0, std::int32_t len) {
        te::NormArgs a;
        a.x = x + static_cast<std::int64_t>(s0) * kTeHidden;
        a.out = out + static_cast<std::int64_t>(s0) * kTeHidden;
        a.ldx = a.ldo = kTeHidden;
        a.weight = weight;
        a.tokens = len;
        a.d = kTeHidden;
        te::rms_norm(a, s);
      });
    });
  };

  // GEMMs: the input packed once for the GEMMs that read it (packed path), then each projection group as one GEMM.
  const auto pack_input = [&](const bf16* in, std::int64_t ld, int k) {
    if (packed_path) op("pack activations", [&] { hip_block::pack_fragments(xp, in, ld, T, te::pack_rows(T), k, s); });
  };
  const auto linear = [&](const char* name, bf16* out, std::int64_t ldo, const bf16* in, std::int64_t ldi, const bf16* w,
                          const Gemm& g) {
    op(name, [&] {
      if (options.tiled_gemm) {
        bf16_gemm_tiled(out, ldo, in, ldi, w + g.raw / 2, g.k, static_cast<int>(T), g.n, g.k, s);
      } else if (packed_path) {
        te::gemm_packed(out, ldo, xp, w + g.packed / 2, static_cast<int>(T), g.n, g.k, s);
      } else {
        hip_block::gemm(out, ldo, in, ldi, w + g.raw / 2, g.k, static_cast<int>(T), g.n, g.k, s);
      }
    });
  };
  const auto linear_residual = [&](const char* name, const bf16* in, std::int64_t ldi, const bf16* w, const Gemm& g) {
    op(name, [&] {
      if (options.tiled_gemm) {
        bf16_gemm_tiled(x, kTeHidden, in, ldi, w + g.raw / 2, g.k, static_cast<int>(T), g.n, g.k, s, x);
      } else if (packed_path) {
        te::gemm_packed_residual(x, kTeHidden, xp, w + g.packed / 2, ones, static_cast<int>(T), g.n, g.k, s);
      } else {
        hip_block::gemm_gated_residual(x, kTeHidden, in, ldi, w + g.raw / 2, g.k, ones, static_cast<int>(T), g.n, g.k, s);
      }
    });
  };

  const auto run_layer = [&](const LayerPlan& l, const bf16* w) {
    const auto W = [&](std::uint64_t offset) { return w + offset / 2; };
    hidden_norm(W(l.in_norm), h);
    pack_input(h, kTeHidden, kTeHidden);
    if (l.full_attention) {
      bf16 *const qkv = at(sc.qkv), *const q = at(sc.q), *const attn = at(sc.attn);
      linear("gemm q|k|v", qkv, kQkvDim, h, kTeHidden, w, l.qkv);
      op("q, k norm+rope", [&] {
        for_each_sequence([&](std::int32_t s0, std::int32_t len) {
          te::NormArgs qn;
          qn.x = qkv + static_cast<std::int64_t>(s0) * kQkvDim;
          qn.ldx = kQkvDim;
          qn.hsx = 2 * kTeHeadDim;
          qn.out = q + static_cast<std::int64_t>(s0) * kTeHidden;
          qn.ldo = kTeHidden;
          qn.hso = kTeHeadDim;
          qn.weight = W(l.q_norm);
          qn.rope = rope_d;
          qn.rope_ld = te::kRopeDim;
          qn.tokens = len;
          qn.heads = kTeHeads;
          qn.d = kTeHeadDim;
          te::rms_norm(qn, s);
          te::NormArgs kn = qn;
          kn.x = kn.out = qkv + static_cast<std::int64_t>(s0) * kQkvDim + kQgDim;
          kn.ldx = kn.ldo = kQkvDim;
          kn.hsx = kn.hso = kTeHeadDim;
          kn.weight = W(l.k_norm);
          kn.heads = kTeKvHeads;
          te::rms_norm(kn, s);
        });
      });
      op("attention+gate", [&] {
        te::AttentionArgs a;
        a.q = q;
        a.k = qkv + kQgDim;
        a.v = qkv + kQgDim + kKvDim;
        a.o = attn;
        a.gate = qkv + kTeHeadDim;
        a.ldq = a.ldo = kTeHidden;
        a.ldk = a.ldv = a.ldg = kQkvDim;
        a.hsq = a.hsk = a.hsv = a.hso = kTeHeadDim;
        a.hsg = 2 * kTeHeadDim;
        a.cu = cu_d;
        a.batch = nseq;
        a.max_seqlen = static_cast<int>(max_len);
        a.heads = kTeHeads;
        a.kv_group = kTeHeads / kTeKvHeads;
        a.scale = 0.0625f;
        te::causal_attention(a, s);
      });
      pack_input(attn, kTeHidden, kTeHidden);
      linear_residual("gemm o+residual", attn, kTeHidden, w, l.o);
    } else {
      // One GEMM's rows [qkv(8192) | z(4096) | b(32) | a(32)].
      bf16 *const qkvz = at(sc.qkvzba), *const conv = at(sc.conv), *const beta = at(sc.beta), *const core = at(sc.core);
      const bf16* ba = qkvz + te::kGdnConvDim + te::kGdnValueDim;
      auto* g = reinterpret_cast<float*>(base + sc.g);
      linear("gemm qkv|z|b|a", qkvz, kQkvzbaDim, h, kTeHidden, w, l.qkvzba);
      op("conv1d+silu", [&] { te::conv_silu(qkvz, kQkvzbaDim, W(l.conv), conv, te::kGdnConvDim, cu_d, nseq, T, s); });
      op("gates", [&] { te::gates(ba, kQkvzbaDim, W(l.a_log), W(l.dt_bias), g, beta, T, s); });
      op("gated delta rule", [&] {
        te::gated_delta_rule(conv, te::kGdnConvDim, g, beta, cu_d, nseq, static_cast<int>(max_len), T, core,
                             te::GdnWorkspace::at(base + sc.ws, T), s);
      });
      op("gated rms_norm", [&] {
        for_each_sequence([&](std::int32_t s0, std::int32_t len) {
          te::NormArgs a;
          a.x = a.out = core + static_cast<std::int64_t>(s0) * te::kGdnValueDim;
          a.ldx = a.ldo = te::kGdnValueDim;
          a.hsx = a.hso = te::kGdnHeadDim;
          a.gate = qkvz + static_cast<std::int64_t>(s0) * kQkvzbaDim + te::kGdnConvDim;
          a.ldg = kQkvzbaDim;
          a.hsg = te::kGdnHeadDim;
          a.weight = W(l.gdn_norm);
          a.tokens = len;
          a.heads = te::kGdnValueHeads;
          a.d = te::kGdnHeadDim;
          te::rms_norm(a, s);
        });
      });
      pack_input(core, te::kGdnValueDim, te::kGdnValueDim);
      linear_residual("gemm out+residual", core, te::kGdnValueDim, w, l.out);
    }
    hidden_norm(W(l.post_norm), h);
    pack_input(h, kTeHidden, kTeHidden);
    bf16 *const gu = at(sc.gu), *const mlp = at(sc.act);
    linear("gemm gate|up", gu, kGateUpDim, h, kTeHidden, w, l.gate_up);
    op("silu_mul", [&] { te::silu_mul(gu, kGateUpDim, mlp, kTeIntermediate, T, kTeIntermediate, s); });
    pack_input(mlp, kTeIntermediate, kTeIntermediate);
    linear_residual("gemm down+residual", mlp, kTeIntermediate, w, l.down);
  };

  // The packed path: a layer's small tensors copied and its groups packed from raw[0] into `packed`.
  const auto pack_layer = [&](const LayerPlan& l) {
    op("pack weights", [&] {
      copy_async(packed.data(), raw[0].data(), l.small_bytes, s);
      const bf16* r = raw[0].as<bf16>();
      bf16* p = packed.as<bf16>();
      for (const Gemm* g : {&l.qkvzba, &l.out, &l.qkv, &l.o, &l.gate_up, &l.down}) {
        if (g->n == 0) continue;
        hip_block::pack_fragments(p + g->packed / 2, r + g->raw / 2, g->k, g->n, te::pack_rows(g->n), g->k, s);
      }
    });
  };

  // Everything that can throw before the try below is made before the reader starts: a throw while it runs outside the
  // try would destroy a joinable thread.
  Event copy_done[2], raw_free[2], dequant_free;
  std::vector<std::unique_ptr<Event>> layer_start, layer_end;
  for (int k = 0; k <= kTeLayers; ++k) {
    layer_start.push_back(std::make_unique<Event>(true));
    layer_end.push_back(std::make_unique<Event>(true));
  }
  Handoff handoff;
  std::atomic<std::int64_t> read_ns{0};
  std::thread reader([&] {
    try {
      for (int k = 0; k < kTeLayers; ++k) {
        const int slot = k % 2;
        {
          std::unique_lock<std::mutex> lock(handoff.mu);
          handoff.cv.wait(lock, [&] { return handoff.abort || (!handoff.busy[slot] && handoff.filled[slot] < 0); });
          if (handoff.abort) return;
          handoff.busy[slot] = true;
        }
        const auto t0 = Clock::now();
        read_layer(layers[static_cast<std::size_t>(k)], staging[slot].as<std::byte>());
        read_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
        std::lock_guard<std::mutex> lock(handoff.mu);
        handoff.busy[slot] = false;
        handoff.filled[slot] = k;
        handoff.cv.notify_all();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(handoff.mu);
      handoff.error = std::current_exception();
      handoff.abort = true;
      handoff.cv.notify_all();
    }
  });
  const auto stop_reader = [&] {
    {
      std::lock_guard<std::mutex> lock(handoff.mu);
      handoff.abort = true;
      handoff.cv.notify_all();
    }
    if (reader.joinable()) reader.join();
  };
  const auto wait_staged = [&](int k) {
    const auto t0 = Clock::now();
    std::unique_lock<std::mutex> lock(handoff.mu);
    handoff.cv.wait(lock, [&] { return handoff.error != nullptr || handoff.filled[k % 2] == k; });
    if (handoff.error) std::rethrow_exception(handoff.error);
    stats.read_wait_s += seconds_since(t0);
  };
  const auto release_staged = [&](int k) {
    std::lock_guard<std::mutex> lock(handoff.mu);
    handoff.filled[k % 2] = -1;
    handoff.cv.notify_all();
  };

  // Layer k's staging -> device copy on the copy stream, once that buffer's last reader is done. On the device-dequant
  // path the int8 bytes land in the shared i8dev first and the compute stream expands them into raw[b].
  const auto issue_copy = [&](int k) {
    wait_staged(k);
    const int b = k % nraw;
    if (devdq) {
      if (k >= 1) dequant_free.wait(cs);  // the previous layer's dequant has consumed the shared staging buffer
      copy_async(i8dev.data(), staging[k % 2].data(), layers[static_cast<std::uint64_t>(k)].bytes, cs);
    } else {
      if (k >= nraw) raw_free[b].wait(cs);
      copy_async(raw[b].data(), staging[k % 2].data(), layers[static_cast<std::uint64_t>(k)].bytes, cs);
    }
    copy_done[b].record(cs);
  };

  std::vector<std::vector<bf16>> result;
  try {
    {
      const auto t0 = Clock::now();
      std::vector<bf16> rows(static_cast<std::size_t>(T) * kTeHidden);
      std::size_t r = 0;
      for (const auto& sq : seqs) {
        for (const std::int32_t id : sq) {
          if (embed.file) {
            embed.file->read(*embed.info, rows.data() + r++ * kTeHidden, static_cast<std::uint64_t>(id) * kTeHidden * 2,
                             kTeHidden * 2);
          } else {
            std::memcpy(rows.data() + r++ * kTeHidden, embed.host_bf16 + static_cast<std::size_t>(id) * kTeHidden,
                        kTeHidden * 2);
          }
        }
      }
      act.upload(rows.data(), rows.size() * 2, sc.x);
      // The uploads above are pageable hipMemcpy, whose DMA may still be in flight on return; the streams are
      // non-blocking and would not wait for it.
      device.synchronize();
      stats.embed_s = seconds_since(t0);
    }
    issue_copy(0);
    for (int k = 0; k < kTeLayers; ++k) {
      const LayerPlan& l = layers[static_cast<std::size_t>(k)];
      const int b = k % nraw;
      copy_done[b].wait(s);
      if (devdq) {
        // Copy only the bf16 pieces verbatim; the int8 pieces are expanded directly into their raw positions.
        char* rd = static_cast<char*>(raw[b].data());
        const char* id = static_cast<const char*>(i8dev.data());
        for (const CopyRange& r : l.d2d_ranges) {
          copy_async(rd + r.offset, id + r.offset, static_cast<std::size_t>(r.bytes), s);
        }
        const bf16* ib = i8dev.as<bf16>();
        bf16* rb = raw[b].as<bf16>();
        const float* scd = scales_dev.as<float>();
        for (const I8Piece& p : l.i8_pieces) {
          te_dequant::dequant_i8(reinterpret_cast<const std::int8_t*>(ib + p.offset / 2), scd + p.scale_off,
                                 rb + p.offset / 2, p.n, p.k, s);
        }
        dequant_free.record(s);
      }
      layer_start[static_cast<std::size_t>(k)]->record(s);
      const bf16* w = raw[b].as<bf16>();
      if (packed_path) {
        pack_layer(l);
        raw_free[0].record(s);
        w = packed.as<bf16>();
      }
      run_layer(l, w);
      if (!packed_path) raw_free[b].record(s);
      layer_end[static_cast<std::size_t>(k)]->record(s);
      // Enqueued after the layer's work, so the host's wait for the reader leaves the compute stream busy; on the device
      // the copy still starts as soon as raw_free allows.
      if (k + 1 < kTeLayers) issue_copy(k + 1);
      if (hook) hook(k, x, T, s);
      if (k + 1 < kTeLayers) {
        // The next layer's staging slot is free once its copy is done: the reader then fills it with layer k + 3.
        copy_done[(k + 1) % nraw].synchronize();
        release_staged(k + 1);
      }
      if (k == 0) release_staged(0);  // its copy finished before layer 0 began
    }
    stop_reader();
    layer_start[kTeLayers]->record(s);
    hidden_norm(final_norm_w, h);
    layer_end[kTeLayers]->record(s);
    stream.synchronize();

    std::vector<bf16> out(static_cast<std::size_t>(T) * kTeHidden);
    act.download(out.data(), out.size() * 2, sc.h);
    for (int i = 0; i < nseq; ++i) {
      result.emplace_back(out.begin() + std::ptrdiff_t{cu[static_cast<std::size_t>(i)]} * kTeHidden,
                          out.begin() + std::ptrdiff_t{cu[static_cast<std::size_t>(i) + 1]} * kTeHidden);
    }
    stats.layers_ms = layer_end[kTeLayers]->elapsed_ms(*layer_start[0]);
    for (std::size_t k = 1; k < kTeLayers; ++k) stats.weight_wait_ms += layer_start[k]->elapsed_ms(*layer_end[k - 1]);
    for (const OpTiming& e : op_events) stats.op_ms[e.name] += e.end->elapsed_ms(*e.start);
  } catch (...) {
    stop_reader();
    try {
      copy_stream.synchronize();
      stream.synchronize();
    } catch (...) {  // already failing; the device may be lost
    }
    throw;
  }
  stats.read_s = static_cast<double>(read_ns.load()) * 1e-9;
  stats.total_s = seconds_since(t_start);
  return result;
}

}  // namespace mi::hip
