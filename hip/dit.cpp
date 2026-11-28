#include "dit.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "bf16_gemm.h"
#include "dit_kernels.h"
#include "safetensors.h"
#include "te_dequant.h"
#include "weights/quant/quant_codec.h"

namespace mi::hip {
namespace {

using bf16 = std::uint16_t;
constexpr std::int64_t kDim = kDiTDim, kFfn = kDiTFfnDim, kHead = kDiTHeadDim, kQkv = 3 * kDiTDim;
constexpr std::size_t kBf16 = 2;
constexpr std::size_t kAlign = 256;

// The longest forward the engine runs (dit_block_fused.h's kMaxRunRows): the cond, uncond and SLG rows of a 4.2 MP
// image, each with a 4096-token prompt. MossDit::forward refuses more joint tokens, and so does this one.
constexpr std::uint64_t kMaxRunRows = 3 * (4'200'000 / 256 + 4096);

bf16 to_bf16(float v) {
  const auto u = std::bit_cast<std::uint32_t>(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<bf16>((u >> 16) | 0x0040u);
  return static_cast<bf16>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(bf16 b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("hip dit: " + what); }

std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// A [rows, cols] row-major bf16 matrix into hip_block::pack_fragments' layout: 16 x 16 fragments contiguous, row groups
// of 16 rows, column groups of 16 columns. rows and cols multiples of 16.
void pack_host(bf16* to, const bf16* from, std::int64_t rows, std::int64_t cols) {
  const std::int64_t groups = cols / 16;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t g = 0; g < groups; ++g) std::memcpy(to + ((r / 16 * groups + g) * 256 + r % 16 * 16), from + r * cols + g * 16, 32);
  }
}

bf16* at(const DeviceBuffer& b, std::size_t byte_offset) {
  return reinterpret_cast<bf16*>(static_cast<char*>(b.data()) + byte_offset);
}

// One block's tensors: the norms, then the GEMM weights packed, q, k and v back to back as the [9216, 3072] QKV weight.
struct BlockLayout {
  std::size_t norm1, norm2, q_norm, k_norm, qkv, o, up_gate, down, bytes;
};
constexpr BlockLayout block_layout() {
  BlockLayout l{};
  std::size_t o = 0;
  const auto place = [&](std::size_t bytes) {
    const std::size_t at_ = o;
    o += (bytes + kAlign - 1) / kAlign * kAlign;
    return at_;
  };
  l.norm1 = place(kDim * kBf16);
  l.norm2 = place(kDim * kBf16);
  l.q_norm = place(kHead * kBf16);
  l.k_norm = place(kHead * kBf16);
  l.qkv = place(kQkv * kDim * kBf16);
  l.o = place(kDim * kDim * kBf16);
  l.up_gate = place(2 * kFfn * kDim * kBf16);
  l.down = place(kDim * kFfn * kBf16);
  l.bytes = o;
  return l;
}
constexpr BlockLayout kBlock = block_layout();

// The INT8 ConvRot block layout: the bf16 norms, then each linear's int8 weight [n, k] and its per-row float scale [n].
struct Int8Layout {
  std::size_t qkv_q, qkv_scale, o_q, o_scale, up_q, up_scale, down_q, down_scale, bytes;
};
constexpr Int8Layout int8_layout() {
  Int8Layout l{};
  const std::size_t norms = 2 * kDim * kBf16 + 2 * kHead * kBf16;
  l.qkv_q = norms;
  l.qkv_scale = l.qkv_q + 3 * kDim * kDim;
  l.o_q = l.qkv_scale + 3 * kDim * 4;
  l.o_scale = l.o_q + kDim * kDim;
  l.up_q = l.o_scale + kDim * 4;
  l.up_scale = l.up_q + 2 * kFfn * kDim;
  l.down_q = l.up_scale + 2 * kFfn * 4;
  l.down_scale = l.down_q + kDim * kFfn;
  l.bytes = l.down_scale + kDim * 4;
  return l;
}
constexpr Int8Layout kInt8 = int8_layout();

// Rebuild the block's pointers from a device base (a resident block or a streaming slot).
Int8BlockWeights int8_weights_at(const char* base) {
  const auto fp = [&](std::size_t o) { return reinterpret_cast<const float*>(base + o); };
  Int8BlockWeights w;
  w.norm1 = reinterpret_cast<const bf16*>(base);
  w.norm2 = reinterpret_cast<const bf16*>(base + kDim * kBf16);
  w.q_norm = reinterpret_cast<const bf16*>(base + 2 * kDim * kBf16);
  w.k_norm = reinterpret_cast<const bf16*>(base + 2 * kDim * kBf16 + kHead * kBf16);
  w.qkv.q = reinterpret_cast<const std::int8_t*>(base + kInt8.qkv_q);
  w.qkv.scale = fp(kInt8.qkv_scale);
  w.qkv.bias = nullptr;
  w.o.q = reinterpret_cast<const std::int8_t*>(base + kInt8.o_q);
  w.o.scale = fp(kInt8.o_scale);
  w.o.bias = nullptr;
  w.up_gate.q = reinterpret_cast<const std::int8_t*>(base + kInt8.up_q);
  w.up_gate.scale = fp(kInt8.up_scale);
  w.up_gate.bias = nullptr;
  w.down.q = reinterpret_cast<const std::int8_t*>(base + kInt8.down_q);
  w.down.scale = fp(kInt8.down_scale);
  w.down.bias = nullptr;
  return w;
}

// The shapes of the tensors the constructor loads into its two kinds of resident buffers, in their order: the globals
// (embedders, time embedding, modulations, head, the refiner's projection), and each of the two refiner blocks.
std::vector<std::vector<std::int64_t>> global_shapes() {
  const std::int64_t d = kDim;
  return {{d, kDiTLatentChannels}, {d}, {d, kDiTFreqDim}, {d}, {d, d}, {d}, {6 * d, d}, {6 * d}, {2 * d, d}, {2 * d},
          {d}, {kDiTLatentChannels, d}, {d, kDiTTextDim}, {d}};
}
std::vector<std::vector<std::int64_t>> text_block_shapes() {
  const std::int64_t d = kDim, f = kFfn;
  return {{d}, {d}, {kHead}, {kHead}, {3 * d, d}, {3 * d}, {d, d}, {d}, {f, d}, {f, d}, {f}, {f}, {d, f}, {d}};
}
// A buffer of bf16 tensors of these shapes, each at a kAlign-aligned offset.
std::uint64_t buffer_bytes(const std::vector<std::vector<std::int64_t>>& shapes) {
  std::uint64_t total = 0;
  for (const auto& s : shapes) {
    std::uint64_t n = kBf16;
    for (const std::int64_t e : s) n *= static_cast<std::uint64_t>(e);
    total += aligned(n);
  }
  return total;
}

std::size_t numel(const std::vector<std::int64_t>& shape) {
  std::size_t n = 1;
  for (std::int64_t d : shape) n *= static_cast<std::size_t>(d);
  return n;
}

}  // namespace

std::uint64_t Dit::block_bytes() { return kBlock.bytes; }

std::uint64_t Dit::minimum_weight_bytes() {
  return buffer_bytes(global_shapes()) + 2 * buffer_bytes(text_block_shapes()) + 2 * kBlock.bytes;
}

int Dit::resident_blocks_for(std::uint64_t device_budget) {
  if (device_budget == 0) return kDiTLayers;
  const std::uint64_t globals = buffer_bytes(global_shapes()) + 2 * buffer_bytes(text_block_shapes());
  const std::uint64_t free = device_budget > globals ? device_budget - globals : 0;
  // Streamed blocks need two slots: the next one copies while the current one runs.
  return free >= kDiTLayers * kBlock.bytes ? kDiTLayers
                                           : static_cast<int>(free > 2 * kBlock.bytes ? free / kBlock.bytes - 2 : 0);
}

Dit::Dit(Device& device, const std::filesystem::path& transformer_dir, const DitOptions& options)
    : device_(device), options_(options) {
  if (std::getenv("MI_HIP_OP_TIME") != nullptr) options_.time_ops = true;  // perf diagnostics, off by default
  device_.make_current();
  const st::Index index(transformer_dir);
  const auto check_tensor = [&](const std::string& name, const std::vector<std::int64_t>& shape) {
    const st::TensorInfo& t = index.at(name);
    if (t.dtype != "BF16") fail(name + " is " + t.dtype + ", expected BF16");
    if (t.shape != shape) fail(name + " has an unexpected shape");
    return t.bytes;
  };
  // StarNodes-style INT8 ConvRot files quantize every linear, including the globals and the text refiner, which this
  // backend runs as bf16. Their `_quantization_metadata` says which modules to undo; the DiT blocks stay INT8.
  const std::string* qmd = index.metadata("_quantization_metadata");
  const auto convrot = st::convrot_groups(qmd != nullptr ? *qmd : "");
  // Tensors into one device buffer, each at a 256-byte aligned offset. BF16 tensors are read into one host staging
  // area and uploaded once per buffer; INT8 ConvRot extras share one q/scale staging + scratch area and are all
  // expanded on the device into their final bf16 slots in one queued burst. This used to do ~5 device-wide syncs
  // and two allocations per extra linear (~75 syncs for 15 linears); now it is five syncs and one scratch per load.
  const auto load = [&](const std::vector<std::string>& names, const std::vector<std::vector<std::int64_t>>& shapes) {
    std::vector<std::size_t> offsets;
    std::size_t total = 0;
    for (std::size_t i = 0; i < names.size(); ++i) {
      offsets.push_back(total);
      total += aligned(numel(shapes[i]) * kBf16);
    }
    owned_.emplace_back(total);
    DeviceBuffer& buf = owned_.back();

    struct Extra {
      std::string name, sname;
      std::size_t q_off, s_off, s_bytes, out_off;
      int group;
      std::size_t rows, cols;
    };
    std::vector<Extra> extras;
    std::size_t qs_bytes = 0;
    for (std::size_t i = 0; i < names.size(); ++i) {
      const st::TensorInfo& t = index.at(names[i]);
      if (t.shape != shapes[i]) fail(names[i] + " has an unexpected shape");
      if (t.dtype != "I8") continue;
      if (shapes[i].size() != 2 || !names[i].ends_with(".weight")) fail(names[i] + " is I8 but not a weight matrix");
      const std::string module = names[i].substr(0, names[i].size() - 7);
      const auto it = convrot.find(module);
      if (it == convrot.end() || !quant::valid_convrot_group(it->second) || shapes[i][1] % it->second != 0)
        fail(names[i] + " is INT8 but has no int8_convrot config in _quantization_metadata");
      const std::string sname = names[i] + "_scale";
      const st::TensorInfo& scale = index.at(sname);
      // Comfy-kitchen writes [N, 1]; the converter's own files use [N]. Both are one scale per output row.
      const bool scales_ok = scale.dtype == "F32" && (scale.shape == std::vector<std::int64_t>{shapes[i][0]} ||
                                                      scale.shape == std::vector<std::int64_t>({shapes[i][0], 1}));
      if (!scales_ok) fail(sname + " is not F32 [N] or [N, 1] as INT8 ConvRot requires");
      extras.push_back({names[i], sname, qs_bytes, qs_bytes + t.bytes, scale.bytes, offsets[i], it->second,
                        static_cast<std::size_t>(shapes[i][0]), static_cast<std::size_t>(shapes[i][1])});
      qs_bytes += t.bytes + scale.bytes;
    }

    std::vector<std::byte> qs(qs_bytes);
    for (const Extra& e : extras) {
      index.read_into(e.name, qs.data() + e.q_off, e.s_off - e.q_off);
      index.read_into(e.sname, qs.data() + e.s_off, e.s_bytes);
    }
    std::vector<std::byte> host(total);
    for (std::size_t i = 0; i < names.size(); ++i) {
      const st::TensorInfo& t = index.at(names[i]);
      if (t.dtype == "BF16") index.read_into(names[i], host.data() + offsets[i], t.bytes);
    }
    // The text encoder's g=256 butterfly is accurate to <0.05 % of bf16 weight bits versus quant::dequantize_rows
    // (one ulp at most), so use that fast kernel here too. Other groups are rare; do them on the host into the same
    // staging buffer so they cost no extra upload or sync.
    for (const Extra& e : extras) {
      if (e.group == 256 && e.cols % 256 == 0) continue;
      std::vector<float> deq(e.rows * e.cols);
      quant::dequantize_rows(reinterpret_cast<const std::int8_t*>(qs.data() + e.q_off),
                             reinterpret_cast<const float*>(qs.data() + e.s_off), static_cast<std::int64_t>(e.rows),
                             static_cast<std::int64_t>(e.cols), e.group, deq.data());
      std::uint16_t* dst = reinterpret_cast<std::uint16_t*>(host.data() + e.out_off);
      for (std::size_t i = 0; i < deq.size(); ++i) dst[i] = quant::float_to_bf16(deq[i]);
    }
    buf.upload(host.data(), total, 0);
    if (!extras.empty()) {
      DeviceBuffer qdev(qs_bytes);
      qdev.upload(qs.data(), qs_bytes);
      for (const Extra& e : extras) {
        if (e.group != 256 || e.cols % 256 != 0) continue;
        te_dequant::dequant_i8(reinterpret_cast<const std::int8_t*>(qdev.data()) + e.q_off,
                               reinterpret_cast<const float*>(qdev.data()) + e.s_off / sizeof(float),
                               reinterpret_cast<std::uint16_t*>(at(buf, e.out_off)), static_cast<int>(e.rows),
                               static_cast<int>(e.cols), stream_.get());
      }
      stream_.synchronize();
    }
    weight_bytes_ += total;
    std::vector<const bf16*> out;
    for (const std::size_t o : offsets) out.push_back(at(buf, o));
    return out;
  };
  const std::int64_t d = kDim, f = kFfn;

  const auto a = load({"video_embedder.weight", "video_embedder.bias", "time_embedding.0.weight", "time_embedding.0.bias",
                       "time_embedding.2.weight", "time_embedding.2.bias", "shared_block_modulation.weight",
                       "shared_block_modulation.bias", "video_head_modulation.weight", "video_head_modulation.bias",
                       "final_norm_video.weight", "final_linear_video.weight", "text_embedder.proj.weight",
                       "text_embedder.proj.bias"},
                      global_shapes());
  g_ = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]};

  // The refiner's ffn_up and ffn_gate (and their biases) back to back form the [hidden | gate] weight of one GEMM, as the
  // engine fuses them.
  for (int b = 0; b < 2; ++b) {
    const std::string p = "text_embedder.blocks." + std::to_string(b) + ".";
    const auto t = load({p + "norm1.weight", p + "norm2.weight", p + "q_norm.weight", p + "k_norm.weight", p + "qkv.weight",
                         p + "qkv.bias", p + "proj.weight", p + "proj.bias", p + "ffn_up.weight", p + "ffn_gate.weight",
                         p + "ffn_up.bias", p + "ffn_gate.bias", p + "ffn_down.weight", p + "ffn_down.bias"},
                        text_block_shapes());
    if (t[9] != t[8] + f * d || t[11] != t[10] + f) fail("the refiner's ffn_up and ffn_gate are not contiguous");
    text_blocks_.push_back({t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[10], t[12], t[13]});
  }

  // The blocks: the first `resident` on the device, the rest in pinned host memory, all packed on the host.
  // The official bf16 bundle keeps proj.q/k/v.weight apart (the engine fuses them at load); the converted
  // INT8 ConvRot bundle stores one proj.qkv.weight. Accept both so --tier reference works with either file.
  const bool fused_qkv = index.has("layers.0.proj.qkv.weight");
  const auto qkv_names = [&](int l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    return fused_qkv ? std::vector<std::string>{p + "proj.qkv.weight"}
                     : std::vector<std::string>{p + "proj.q.weight", p + "proj.k.weight", p + "proj.v.weight"};
  };
  const auto block_names = [](int l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    return std::vector<std::string>{p + "norm1.weight", p + "norm2.weight", p + "q_norm.weight", p + "k_norm.weight",
                                    p + "proj.o.weight", p + "proj.ffn_up_gate.weight", p + "proj.ffn_down.weight"};
  };
  const std::vector<std::vector<std::int64_t>> block_shapes = {{d}, {d}, {kHead}, {kHead}, {d, d}, {2 * f, d},
                                                               {d, f}};
  // The globals and the refiner stay resident, and streaming needs its two slots: a budget below that cannot hold.
  if (options.device_budget > 0 && options.device_budget < minimum_weight_bytes()) {
    fail("a device budget of " + std::to_string(options.device_budget) + " bytes is below the " +
         std::to_string(minimum_weight_bytes()) +
         " the DiT takes at least (its embedders, modulations and text refiner, and two block slots)");
  }
  const int resident = resident_blocks_for(options.device_budget);
  HostBuffer staging(resident > 0 ? kBlock.bytes : 0);
  std::vector<bf16> row_major;
  is_int8_.assign(kDiTLayers, false);
  int8_blocks_.resize(kDiTLayers);
  // INT8 blocks use the same resident/streaming split as bf16 blocks. An INT8 block is about half the size of a packed
  // bf16 one, so a budget sized from bf16 blocks is conservative; the slots take the larger of the two layouts so a
  // mixed model can stream either kind.
  const std::size_t i8_block_bytes = kInt8.bytes;
  const std::size_t slot_bytes = std::max(kBlock.bytes, kInt8.bytes);
  for (int l = 0; l < kDiTLayers; ++l) {
    const std::vector<std::string> qkv = qkv_names(l);
    const bool is_int8 = index.at(qkv.front()).dtype == "I8";
    is_int8_[l] = is_int8;
    if (is_int8) {
      const std::string p = "layers." + std::to_string(l) + ".";
      const auto i8_check = [&](const std::string& name, const std::vector<std::int64_t>& shape, const std::string& dt) {
        const st::TensorInfo& t = index.at(name);
        if (t.dtype != dt) fail(name + " is " + t.dtype + ", expected " + dt);
        if (t.shape != shape) fail(name + " has an unexpected shape");
      };
      i8_check(p + "norm1.weight", {d}, "BF16");
      i8_check(p + "norm2.weight", {d}, "BF16");
      i8_check(p + "q_norm.weight", {kHead}, "BF16");
      i8_check(p + "k_norm.weight", {kHead}, "BF16");
      const std::int64_t qrows = fused_qkv ? 3 * d : d;
      for (const std::string& name : qkv) {
        i8_check(name, {qrows, d}, "I8");
        i8_check(name + "_scale", {qrows, 1}, "F32");
      }
      i8_check(p + "proj.o.weight", {d, d}, "I8");
      i8_check(p + "proj.o.weight_scale", {d, 1}, "F32");
      i8_check(p + "proj.ffn_up_gate.weight", {2 * f, d}, "I8");
      i8_check(p + "proj.ffn_up_gate.weight_scale", {2 * f, 1}, "F32");
      i8_check(p + "proj.ffn_down.weight", {d, f}, "I8");
      i8_check(p + "proj.ffn_down.weight_scale", {d, 1}, "F32");
      HostBuffer streamed(i8_block_bytes);
      char* dst = static_cast<char*>(streamed.data());
      index.read_into(p + "norm1.weight", dst + 0, d * kBf16);
      index.read_into(p + "norm2.weight", dst + d * kBf16, d * kBf16);
      index.read_into(p + "q_norm.weight", dst + 2 * d * kBf16, kHead * kBf16);
      index.read_into(p + "k_norm.weight", dst + 2 * d * kBf16 + kHead * kBf16, kHead * kBf16);
      for (std::size_t i = 0; i < qkv.size(); ++i) {
        index.read_into(qkv[i], dst + kInt8.qkv_q + static_cast<std::size_t>(i * qrows * d),
                        static_cast<std::size_t>(qrows * d));
        index.read_into(qkv[i] + "_scale", dst + kInt8.qkv_scale + static_cast<std::size_t>(i * qrows * 4),
                        static_cast<std::size_t>(qrows * 4));
      }
      index.read_into(p + "proj.o.weight", dst + kInt8.o_q, d * d);
      index.read_into(p + "proj.o.weight_scale", dst + kInt8.o_scale, d * 4);
      index.read_into(p + "proj.ffn_up_gate.weight", dst + kInt8.up_q, 2 * f * d);
      index.read_into(p + "proj.ffn_up_gate.weight_scale", dst + kInt8.up_scale, 2 * f * 4);
      index.read_into(p + "proj.ffn_down.weight", dst + kInt8.down_q, d * f);
      index.read_into(p + "proj.ffn_down.weight_scale", dst + kInt8.down_scale, d * 4);
      if (l < resident) {
        owned_.emplace_back(i8_block_bytes);
        owned_.back().upload(streamed.data(), i8_block_bytes);
        weight_bytes_ += i8_block_bytes;
        int8_blocks_[l] = int8_weights_at(reinterpret_cast<const char*>(at(owned_.back(), 0)));
        slot_of_.push_back(-1);
        host_blocks_.emplace_back();  // resident blocks need no pinned host copy
      } else {
        const int s = (l - resident) % 2;
        if (slots_[s].bytes() == 0) {
          slots_[s] = DeviceBuffer(slot_bytes);
          slot_free_[s] = std::make_unique<Event>();
          copy_done_[s] = std::make_unique<Event>();
          weight_bytes_ += slot_bytes;
        }
        host_weight_bytes_ += i8_block_bytes;
        slot_of_.push_back(s);
      }
      // Keep blocks_ aligned with the layer index: a model with mixed INT8/bf16 blocks indexes this vector by l.
      blocks_.push_back(hip_block::BlockWeights{});
      if (l >= resident) host_blocks_.push_back(std::move(streamed));
      continue;
    }
    const std::vector<std::string> names = block_names(l);
    for (std::size_t i = 0; i < names.size(); ++i) check_tensor(names[i], block_shapes[i]);
    const std::int64_t qrows = fused_qkv ? 3 * d : d;
    for (const std::string& name : qkv) check_tensor(name, {qrows, d});
    const bool on_device = l < resident;
    HostBuffer streamed;
    if (!on_device) streamed = HostBuffer(kBlock.bytes);
    char* dst = static_cast<char*>(on_device ? staging.data() : streamed.data());
    std::memset(dst, 0, kBlock.bytes);
    const std::size_t plain[4] = {kBlock.norm1, kBlock.norm2, kBlock.q_norm, kBlock.k_norm};
    for (std::size_t i = 0; i < 4; ++i) index.read_into(names[i], dst + plain[i], index.at(names[i]).bytes);
    // qkv is stored as [3d, d] = q, k, v stacked (whether it came from one fused tensor or three separate ones);
    // packing it whole gives the [9216, 3072] qkv layout. o, up_gate and down pack to their slots.
    row_major.resize(static_cast<std::size_t>(3 * d * d));
    bf16* qkv_rm = row_major.data();
    for (std::size_t i = 0; i < qkv.size(); ++i) {
      std::vector<bf16> part(static_cast<std::size_t>(qrows * d));
      index.read_into(qkv[i], part.data(), part.size() * kBf16);
      std::memcpy(qkv_rm + static_cast<std::size_t>(i * d * d), part.data(), part.size() * kBf16);
    }
    pack_host(reinterpret_cast<bf16*>(dst + kBlock.qkv), qkv_rm, 3 * d, d);
    const std::size_t packed_at[3] = {kBlock.o, kBlock.up_gate, kBlock.down};
    for (std::size_t i = 4; i < names.size(); ++i) {
      const std::vector<std::int64_t>& s = block_shapes[i];
      row_major.resize(static_cast<std::size_t>(s[0] * s[1]));
      index.read_into(names[i], row_major.data(), row_major.size() * kBf16);
      pack_host(reinterpret_cast<bf16*>(dst + packed_at[i - 4]), row_major.data(), s[0], s[1]);
    }
    const bf16* base = nullptr;
    if (on_device) {
      owned_.emplace_back(kBlock.bytes);
      owned_.back().upload(staging.data(), kBlock.bytes);
      weight_bytes_ += kBlock.bytes;
      base = at(owned_.back(), 0);
      slot_of_.push_back(-1);
    } else {
      const int s = (l - resident) % 2;
      if (slots_[s].bytes() == 0) {
        slots_[s] = DeviceBuffer(slot_bytes);
        slot_free_[s] = std::make_unique<Event>();
        copy_done_[s] = std::make_unique<Event>();
        weight_bytes_ += slot_bytes;
      }
      base = at(slots_[s], 0);
      slot_of_.push_back(s);
      host_weight_bytes_ += kBlock.bytes;
    }
    host_blocks_.push_back(std::move(streamed));
    const auto p = [&](std::size_t o) { return reinterpret_cast<const bf16*>(reinterpret_cast<const char*>(base) + o); };
    hip_block::BlockWeights w{p(kBlock.norm1), p(kBlock.norm2), p(kBlock.q_norm), p(kBlock.k_norm), nullptr, nullptr, nullptr,
                              nullptr};
    w.qkv_packed = p(kBlock.qkv);
    w.o_packed = p(kBlock.o);
    w.up_gate_packed = p(kBlock.up_gate);
    w.down_packed = p(kBlock.down);
    blocks_.push_back(w);
  }
  resident_blocks_ = resident;
}

Dit::~Dit() {
  // Copies in flight read host buffers this destructor frees.
  try {
    copy_stream_.synchronize();
    stream_.synchronize();
  } catch (const std::exception&) {
  }
}

std::uint64_t Dit::activation_bytes(std::uint64_t tokens, std::uint64_t rows, std::uint64_t hw, bool int8, bool lora) {
  // x, y, attn [T, 3072]; qkv [T, 9216]; ffn [T, 8192]; rope [T, 128]; the packed GEMM input [T rounded up to 96, 8192]
  // and the packed attention input [T + 32 rows, 9216]; the video tokens [hw, 128] and each row's projection; the time
  // and modulation vectors.
  const std::uint64_t pack_rows = (tokens + hip_block::kPackRowsMultiple - 1) / hip_block::kPackRowsMultiple * hip_block::kPackRowsMultiple;
  std::uint64_t bytes = (tokens * (3 * kDim + kQkv + kFfn + kHead) + pack_rows * kFfn +
                         (tokens + hip_block::kAttentionStepN * rows) * kQkv +
                         hw * kDiTLatentChannels * (1 + rows) + 10 * kDim + kDiTFreqDim) *
                        kBf16;
  // The INT8 path adds quantized activations and the up_gate bf16 output; the LoRA side branch adds its low-rank t and
  // the full-size delta fused into the base GEMM. Include them in the planner's accounting so a large image cannot be
  // planned as if those buffers were free.
  if (int8) {
    bytes += tokens * (kDim + 5 * kFfn + 1) + tokens * sizeof(float);
    // The int8 attention's q/k/v, scales and varlen metadata.
    const std::uint64_t padded = (tokens + 63u) / 64u * 64u + 64u * rows;
    const std::uint64_t h = static_cast<std::uint64_t>(hip_block::kHeads) * hip_block::kHeadDim;
    bytes += padded * 3u * h + padded * hip_block::kHeads * sizeof(float) +
             padded / 16u * hip_block::kHeads * sizeof(float) + tokens * sizeof(std::int32_t) +
             ((tokens + 255u) / 256u) * h * sizeof(float);
    // RDNA2's gfx103x attention needs a fp16 copy of V; charge it even on builds without RDNA2 so the budget is the
    // same across the dual-architecture binary.
    bytes += padded * h * 2u;
  }
  if (lora) bytes += tokens * (2 * kFfn) * kBf16;
  return bytes;
}

DeviceBuffer& Dit::scratch(DeviceBuffer& b, std::size_t bytes) {
  if (b.bytes() < bytes) {
    b = DeviceBuffer();
    b = DeviceBuffer(bytes);
  }
  return b;
}

void Dit::mark(int what) {
  if (!options_.time_ops) return;
  if (events_used_ == events_.size()) events_.push_back(std::make_unique<Event>(true));
  event_what_.resize(events_.size());
  event_what_[events_used_] = what;
  events_[events_used_++]->record(stream_.get());
}

const bf16* Dit::compute_lora_delta(const LoraBranch& branch, const bf16* x, int k, int n, int m, hipStream_t stream) {
  if (!branch.active || branch.k != k || branch.n != n || m == 0) return nullptr;
  // t = bf16(x · Aᵀ); delta = bf16(t · Bᵀ); out += delta before the site's epilogue. B's columns already carry the
  // adapter's coefficient (folded in on load), so no separate scaling kernel is needed.
  const int r = branch.rank;
  bf16* t = lora_t32_.as<bf16>();
  bf16* delta = lora_delta_.as<bf16>();
  bf16_gemm_tiled(t, r, x, k, branch.a, k, m, r, k, stream);
  bf16_gemm_tiled(delta, n, t, r, branch.b, r, m, n, r, stream);
  return delta;
}

DeviceBuffer Dit::encode_text(const std::vector<std::uint16_t>& embeds, std::uint32_t length) {
  if (length == 0 || embeds.size() != std::size_t{length} * kDiTTextDim) fail("encode_text: embeds must be [length, 4096]");
  device_.make_current();
  // Work queued earlier may still use the scratch buffers resized below, which a reallocation would free under it.
  stream_.synchronize();
  const std::size_t s = length;
  const hipStream_t st = stream_.get();
  DeviceBuffer& e = scratch(embeds_, s * kDiTTextDim * kBf16);
  e.upload(embeds.data(), embeds.size() * kBf16);
  DeviceBuffer& y = scratch(y_, s * kDim * kBf16);
  DeviceBuffer& qkv = scratch(qkv_, s * kQkv * kBf16);
  DeviceBuffer& attn = scratch(attn_, s * kDim * kBf16);
  DeviceBuffer& up_gate = scratch(up_gate_, s * 2 * kFfn * kBf16);
  DeviceBuffer& ffn = scratch(ffn_, s * kFfn * kBf16);
  DeviceBuffer& rope = scratch(rope_, s * kHead * kBf16);
  DeviceBuffer& cu = scratch(cu_, 2 * sizeof(std::int32_t));
  const std::int32_t cu_host[2] = {0, static_cast<std::int32_t>(length)};
  cu.upload(cu_host, sizeof(cu_host));
  DeviceBuffer out(s * kDim * kBf16);
  const int m = static_cast<int>(length);
  bf16* x = out.as<bf16>();

  hip_block::rope_table(rope.as<bf16>(), 0, 0, m, st);  // the text rows of the joint table (P:251)

  hip_block::gemm_bias(x, kDim, e.as<bf16>(), kDiTTextDim, g_.text_w, kDiTTextDim, g_.text_b, m, kDim, kDiTTextDim, st);  // P:120
  for (const TextBlock& b : text_blocks_) {  // _TextBlock.forward, P:104-110
    hip_block::rms_norm(x, kDim, b.norm1, y.as<bf16>(), kDim, m, kDim, st);
    hip_block::gemm_bias(qkv.as<bf16>(), kQkv, y.as<bf16>(), kDim, b.qkv_w, kDim, b.qkv_b, m, kQkv, kDim, st);
    hip_block::qk_norm_rope(qkv.as<bf16>(), kQkv, m, kDiTHeads, b.q_norm, b.k_norm, rope.as<bf16>(), kHead, st);
    hip_block::attention({qkv.as<bf16>(), qkv.as<bf16>() + kDim, qkv.as<bf16>() + 2 * kDim, attn.as<bf16>(), kQkv, kQkv, kQkv,
                          kDim, kHead, kHead, kHead, kHead, cu.as<std::int32_t>(), cu.as<std::int32_t>(), 1, m, kDiTHeads,
                          static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHead)))},
                         st);
    hip_block::gemm_bias_residual(x, kDim, attn.as<bf16>(), kDim, b.proj_w, kDim, b.proj_b, m, kDim, kDim, st);
    hip_block::rms_norm(x, kDim, b.norm2, y.as<bf16>(), kDim, m, kDim, st);
    hip_block::gemm_bias(up_gate.as<bf16>(), 2 * kFfn, y.as<bf16>(), kDim, b.up_gate_w, kDim, b.up_gate_b, m, 2 * kFfn, kDim, st);
    hip_block::situ_glu(up_gate.as<bf16>(), 2 * kFfn, ffn.as<bf16>(), kFfn, m, kFfn, st);
    hip_block::gemm_bias_residual(x, kDim, ffn.as<bf16>(), kFfn, b.down_w, kFfn, b.down_b, m, kDim, kFfn, st);
  }
  stream_.synchronize();
  return out;
}

void Dit::stage_time(float timestep) {
  // P:232-241: a float64 sinusoid rounded to fp32 then bf16 (the .float() and back are exact); time_modulation() runs the
  // bf16 MLP on it.
  constexpr int kHalf = kDiTFreqDim / 2;
  std::vector<bf16> emb(kDiTFreqDim);
  for (int i = 0; i < kHalf; ++i) {
    const double freq = std::pow(10000.0, -static_cast<double>(i) / kHalf);
    const double phase = static_cast<double>(timestep) * freq;
    emb[static_cast<std::size_t>(i)] = to_bf16(static_cast<float>(std::cos(phase)));
    emb[static_cast<std::size_t>(kHalf + i)] = to_bf16(static_cast<float>(std::sin(phase)));
  }
  scratch(t_in_, kDiTFreqDim * kBf16).upload(emb.data(), emb.size() * kBf16);
  scratch(hidden_, kDim * kBf16);
  scratch(time_, kDim * kBf16);
  scratch(mod_, 6 * kDim * kBf16);
  scratch(head_, 2 * kDim * kBf16);
}

void Dit::time_modulation() {
  DeviceBuffer &t_in = t_in_, &hidden = hidden_, &time = time_, &mod = mod_, &head = head_;
  const hipStream_t st = stream_.get();
  hip_block::gemm_bias(hidden.as<bf16>(), kDim, t_in.as<bf16>(), kDiTFreqDim, g_.time0_w, kDiTFreqDim, g_.time0_b, 1, kDim,
                       kDiTFreqDim, st);
  hip_block::silu(hidden.as<bf16>(), hidden.as<bf16>(), kDim, st);
  hip_block::gemm_bias(time.as<bf16>(), kDim, hidden.as<bf16>(), kDim, g_.time2_w, kDim, g_.time2_b, 1, kDim, kDim, st);
  hip_block::gemm_bias(mod.as<bf16>(), 6 * kDim, time.as<bf16>(), kDim, g_.mod_w, kDim, g_.mod_b, 1, 6 * kDim, kDim, st);
  hip_block::gemm_bias(head.as<bf16>(), 2 * kDim, time.as<bf16>(), kDim, g_.head_w, kDim, g_.head_b, 1, 2 * kDim, kDim, st);
}

std::vector<std::vector<float>> Dit::forward(const std::vector<float>& latents, std::uint32_t height, std::uint32_t width,
                                             float timestep, const std::vector<Row>& rows) {
  const std::uint64_t hw = std::uint64_t{height} * width;
  if (rows.empty() || hw == 0 || latents.size() != hw * kDiTLatentChannels) fail("forward: latents must be [128, h, w]");
  std::vector<std::uint64_t> off{0};
  for (const Row& r : rows) {
    if (r.text == nullptr || r.length == 0 || r.text->bytes() < std::uint64_t{r.length} * kDim * kBf16) {
      fail("forward: a row has no refined text");
    }
    off.push_back(off.back() + hw + r.length);
  }
  const std::uint64_t total = off.back();
  if (total > kMaxRunRows) {
    fail("forward: a forward of " + std::to_string(total) + " joint tokens exceeds " + std::to_string(kMaxRunRows));
  }
  device_.make_current();
  // Work queued earlier may still use the scratch buffers resized below, which a reallocation would free under it.
  stream_.synchronize();
  const hipStream_t st = stream_.get();
  events_used_ = 0;
  op_ms_.clear();

  DeviceBuffer& x = scratch(x_, total * kDim * kBf16);
  scratch(y_, total * kDim * kBf16);
  scratch(qkv_, total * kQkv * kBf16);
  scratch(attn_, total * kDim * kBf16);
  scratch(ffn_, total * kFfn * kBf16);
  const std::uint64_t pack_rows =
      (total + hip_block::kPackRowsMultiple - 1) / hip_block::kPackRowsMultiple * hip_block::kPackRowsMultiple;
  scratch(pack_, pack_rows * kFfn * kBf16);
  scratch(pack_attn_, (total + hip_block::kAttentionStepN * rows.size()) * kQkv * kBf16);
  DeviceBuffer& rope = scratch(rope_, total * kHead * kBf16);
  DeviceBuffer& tokens = scratch(tokens_, hw * kDiTLatentChannels * kBf16);
  DeviceBuffer& proj = scratch(proj_, rows.size() * hw * kDiTLatentChannels * kBf16);
  // The INT8 block's scratch (rotated+quantized activations and their GEMM outputs), sized when any block is INT8.
  if (std::any_of(is_int8_.begin(), is_int8_.end(), [](bool v) { return v; })) {
    scratch(i8_y_, total * kDim);
    scratch(i8_ffn_, total * kFfn);
    scratch(i8_scale_, total * sizeof(float));
    scratch(i8_ug_, total * 2 * kFfn * kBf16);
    const std::size_t padded = int8_attn_padded(static_cast<int>(total), static_cast<int>(rows.size()));
    const std::size_t h = static_cast<std::size_t>(hip_block::kHeads) * hip_block::kHeadDim;
    scratch(i8_q_, padded * h);
    scratch(i8_k_, padded * h);
    scratch(i8_v_, padded * h);
    scratch(i8_q_scale_, padded * hip_block::kHeads * sizeof(float));
    scratch(i8_k_scale_, padded / 16 * hip_block::kHeads * sizeof(float));
    scratch(i8_v_scale_, h * sizeof(float));
    scratch(i8_part_, ((static_cast<std::size_t>(total) + 255) / 256) * h * sizeof(float));
    if (device_is_rdna2())
      scratch(i8_v_fp16_, int8_attn_fp16_v_bytes(static_cast<int>(total), static_cast<int>(rows.size())));
  }
  if (options_.lora != nullptr && options_.lora->loaded()) {
    scratch(lora_t32_, total * options_.lora->max_rank() * kBf16);
    scratch(lora_delta_, total * options_.lora->max_n() * kBf16);
  }

  // The video tokens [hw, 128] in bf16 (the reference's .to(bf16) of the transposed latents), the same for every row.
  std::vector<bf16> host_tokens(hw * kDiTLatentChannels);
  for (std::uint64_t p = 0; p < hw; ++p) {
    for (std::uint32_t c = 0; c < kDiTLatentChannels; ++c) host_tokens[p * kDiTLatentChannels + c] = to_bf16(latents[c * hw + p]);
  }
  tokens.upload(host_tokens.data(), host_tokens.size() * kBf16);

  // Each block's runs: the maximal runs of consecutive rows that execute it, each one varlen batch over its contiguous
  // tokens. Their sequence offsets go to the device once.
  struct Run {
    std::size_t r0, r1, cu_at;
  };
  std::vector<std::vector<Run>> runs(kDiTLayers);
  std::vector<std::int32_t> cu_host;
  for (int l = 0; l < kDiTLayers; ++l) {
    for (std::size_t r0 = 0; r0 < rows.size();) {
      if (rows[r0].skip.test(static_cast<std::size_t>(l))) {
        ++r0;
        continue;
      }
      std::size_t r1 = r0;
      while (r1 < rows.size() && !rows[r1].skip.test(static_cast<std::size_t>(l))) ++r1;
      const std::size_t at_ = cu_host.size();
      for (std::size_t r = r0; r <= r1; ++r) cu_host.push_back(static_cast<std::int32_t>(off[r] - off[r0]));
      runs[static_cast<std::size_t>(l)].push_back({r0, r1, at_});
      r0 = r1;
    }
  }
  DeviceBuffer& cu = scratch(cu_, cu_host.size() * sizeof(std::int32_t));
  cu.upload(cu_host.data(), cu_host.size() * sizeof(std::int32_t));
  stage_time(timestep);

  // Every host upload is done: from here on the host only queues work (time_ops measures the device from here).
  mark(-1);
  time_modulation();
  for (std::size_t r = 0; r < rows.size(); ++r) {
    hip_block::rope_table(at(rope, off[r] * kHead * kBf16), static_cast<int>(height), static_cast<int>(width),
                          static_cast<int>(rows[r].length), st);
  }
  hip_block::gemm_bias(at(x, 0), kDim, tokens.as<bf16>(), kDiTLatentChannels, g_.video_w, kDiTLatentChannels, g_.video_b,
                       static_cast<int>(hw), kDim, kDiTLatentChannels, st);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    if (r > 0) copy_async(at(x, off[r] * kDim * kBf16), at(x, 0), hw * kDim * kBf16, st);
    copy_async(at(x, (off[r] + hw) * kDim * kBf16), rows[r].text->data(), std::uint64_t{rows[r].length} * kDim * kBf16, st);
  }
  mark(-1);

  for (int l = 0; l < kDiTLayers; ++l) {
    const int s = slot_of_[static_cast<std::size_t>(l)];
    if (s >= 0) {
      // The copy waits until the slot's previous block has run; this block's ops wait for the copy. The host queues
      // everything ahead, so the next streamed block's copy (the other slot) runs while this block computes.
      slot_free_[s]->wait(copy_stream_.get());
      const HostBuffer& src = host_blocks_[static_cast<std::size_t>(l)];
      copy_async(slots_[s].data(), src.data(), src.bytes(), copy_stream_.get());
      copy_done_[s]->record(copy_stream_.get());
      copy_done_[s]->wait(st);
      if (is_int8_[static_cast<std::size_t>(l)])
        int8_blocks_[static_cast<std::size_t>(l)] = int8_weights_at(reinterpret_cast<const char*>(slots_[s].data()));
      mark(hip_block::kBlockOps);
    }
    const LoraBlock* lb = options_.lora != nullptr && options_.lora->loaded()
                              ? &options_.lora->block(l)
                              : nullptr;
    for (const Run& run : runs[static_cast<std::size_t>(l)]) {
      hip_block::BlockBuffers b{};
      b.x = at(x, off[run.r0] * kDim * kBf16);
      b.y = y_.as<bf16>();
      b.qkv = qkv_.as<bf16>();
      b.attn = attn_.as<bf16>();
      b.ffn = ffn_.as<bf16>();
      b.rope = at(rope, off[run.r0] * kHead * kBf16);
      b.mod = mod_.as<bf16>();
      b.cu_seqlens = cu.as<std::int32_t>() + run.cu_at;
      b.batch = static_cast<int>(run.r1 - run.r0);
      b.max_seqlen = 0;
      for (std::size_t r = run.r0; r < run.r1; ++r) b.max_seqlen = std::max(b.max_seqlen, static_cast<int>(off[r + 1] - off[r]));
      b.tokens = static_cast<int>(off[run.r1] - off[run.r0]);
      b.pack = pack_.as<bf16>();
      b.pack_attn = pack_attn_.as<bf16>();
      if (is_int8_[static_cast<std::size_t>(l)]) {
        Int8Buffers ib{};
        ib.i8_y = i8_y_.as<std::int8_t>();
        ib.i8_ffn = i8_ffn_.as<std::int8_t>();
        ib.scale = i8_scale_.as<float>();
        ib.ug = i8_ug_.as<bf16>();
        ib.attn.q = i8_q_.as<std::int8_t>();
        ib.attn.k = i8_k_.as<std::int8_t>();
        ib.attn.v = i8_v_.as<std::int8_t>();
        ib.attn.q_scale = i8_q_scale_.as<float>();
        ib.attn.k_scale = i8_k_scale_.as<float>();
        ib.attn.v_scale = i8_v_scale_.as<float>();
        ib.attn.part = i8_part_.as<float>();
        ib.attn.v_fp16 = i8_v_fp16_.as<std::uint16_t>();
        ib.attn.cu = cu.as<std::int32_t>() + run.cu_at;
        ib.attn.batch = b.batch;
        ib.attn.padded = static_cast<int>(int8_attn_padded(static_cast<int>(total),
                                                          static_cast<int>(rows.size())));
        for (int op = 0; op < hip_block::kBlockOps; ++op) {
          const bf16* lora_delta = nullptr;
          std::int64_t ldl = 0;
          if (lb != nullptr) {
            if (op == 1) {
              lora_delta = compute_lora_delta(lb->site[static_cast<int>(LoraSite::kQkv)], b.y, kDim, 3 * kDim, b.tokens, st);
              ldl = 3 * kDim;
            } else if (op == 4) {
              lora_delta = compute_lora_delta(lb->site[static_cast<int>(LoraSite::kO)], b.attn, kDim, kDim, b.tokens, st);
              ldl = kDim;
            } else if (op == 6) {
              lora_delta = compute_lora_delta(lb->site[static_cast<int>(LoraSite::kUpGate)], b.y, kDim, 2 * kFfn, b.tokens, st);
              ldl = 2 * kFfn;
            } else if (op == 7) {
              lora_delta = compute_lora_delta(lb->site[static_cast<int>(LoraSite::kDown)], b.ffn, kFfn, kDim, b.tokens, st);
              ldl = kDim;
            }
          }
          run_int8_block_op(op, int8_blocks_[static_cast<std::size_t>(l)], b, ib, st, lora_delta, ldl, lb == nullptr);
          mark(op);
        }
      } else {
        for (int op = 0; op < hip_block::kBlockOps; ++op) {
          const bf16* lora_delta = nullptr;
          std::int64_t ldl = 0;
          if (lb != nullptr) {
            if (op == 1) {
              lora_delta = compute_lora_delta(lb->site[static_cast<int>(LoraSite::kQkv)], b.y, kDim, 3 * kDim, b.tokens,
                                              st);
              ldl = 3 * kDim;
            } else if (op == 4) {
              lora_delta = compute_lora_delta(lb->site[static_cast<int>(LoraSite::kO)], b.attn, kDim, kDim, b.tokens, st);
              ldl = kDim;
            } else if (op == 6) {
              lora_delta = compute_lora_delta(lb->site[static_cast<int>(LoraSite::kUpGate)], b.y, kDim, 2 * kFfn,
                                              b.tokens, st);
              ldl = 2 * kFfn;
            } else if (op == 7) {
              lora_delta = compute_lora_delta(lb->site[static_cast<int>(LoraSite::kDown)], b.ffn, kFfn, kDim, b.tokens,
                                              st);
              ldl = kDim;
            }
          }
          hip_block::run_block_op(op, blocks_[static_cast<std::size_t>(l)], b, st, lora_delta, ldl);
          mark(op);
        }
      }
    }
    if (s >= 0) slot_free_[s]->record(st);
  }

  // Head (P:256-259): final norm with the head modulation (shift = head[0], scale = head[1]), linear, transpose.
  for (std::size_t r = 0; r < rows.size(); ++r) {
    hip_block::rms_norm_modulate(at(x, off[r] * kDim * kBf16), kDim, g_.final_norm, at(head_, kDim * kBf16), head_.as<bf16>(),
                                 y_.as<bf16>(), kDim, static_cast<std::int64_t>(hw), kDim, st);
    hip_block::gemm(at(proj, r * hw * kDiTLatentChannels * kBf16), kDiTLatentChannels, y_.as<bf16>(), kDim, g_.final_w, kDim,
                    static_cast<int>(hw), kDiTLatentChannels, kDim, st);
  }
  mark(-1);
  stream_.synchronize();
  std::vector<bf16> host_proj(rows.size() * hw * kDiTLatentChannels);
  proj.download(host_proj.data(), host_proj.size() * kBf16);
  std::vector<std::vector<float>> out(rows.size(), std::vector<float>(hw * kDiTLatentChannels));
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const bf16* pr = host_proj.data() + r * hw * kDiTLatentChannels;
    for (std::uint64_t p = 0; p < hw; ++p) {
      for (std::uint32_t c = 0; c < kDiTLatentChannels; ++c) out[r][c * hw + p] = from_bf16(pr[p * kDiTLatentChannels + c]);
    }
  }
  if (options_.time_ops) {
    // Per op kind, then the waits for streamed copies, then the rest (time embedding, embedders, head) as the last entry.
    op_ms_.assign(hip_block::kBlockOps + 2, 0.0);
    for (std::size_t i = 1; i < events_used_; ++i) {
      const int what = event_what_[i];
      const double ms = static_cast<double>(events_[i]->elapsed_ms(*events_[i - 1]));
      op_ms_[static_cast<std::size_t>(what < 0 ? hip_block::kBlockOps + 1 : what)] += ms;
    }
    if (std::getenv("MI_HIP_OP_TIME") != nullptr) {
      std::fprintf(stderr, "[op-time]");
      for (std::size_t i = 0; i < op_ms_.size(); ++i) std::fprintf(stderr, " %zu=%.0fms", i, op_ms_[i]);
      std::fprintf(stderr, "\n");
    }
  }
  return out;
}

}  // namespace mi::hip
