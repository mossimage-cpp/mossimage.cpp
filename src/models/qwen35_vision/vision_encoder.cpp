#include "models/qwen35_vision/vision_encoder.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/vision/vision.h"
#include "util/xxh64.h"
#include "weights/safetensors.h"

namespace mi::qwen35_vision {
namespace {

std::uint16_t to_bf16(float v) {
  const auto u = std::bit_cast<std::uint32_t>(v);
  if (std::isnan(v)) return static_cast<std::uint16_t>((u >> 16) | 0x40);
  return static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

float from_bf16(std::uint16_t v) { return std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16); }

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("qwen35_vision: " + what); }

// torch.linspace(start, end, steps) on the CPU in fp32: the first half counts up from start, the second down from end.
std::vector<float> linspace(float start, float end, int steps) {
  std::vector<float> out(static_cast<std::size_t>(steps));
  if (steps == 1) {
    out[0] = start;
    return out;
  }
  const float step = (end - start) / static_cast<float>(steps - 1);
  const int half = steps / 2;
  for (int i = 0; i < steps; ++i) {
    out[static_cast<std::size_t>(i)] = i < half ? start + step * static_cast<float>(i)
                                                : end - step * static_cast<float>(steps - 1 - i);
  }
  return out;
}

// Patch (r, c) of a grid_h x grid_w grid in the merger's block order (2x2 blocks, row-major blocks).
template <typename F>
void for_each_block_major(int grid_h, int grid_w, F&& f) {
  std::int64_t i = 0;
  for (int bh = 0; bh < grid_h / kMerge; ++bh)
    for (int bw = 0; bw < grid_w / kMerge; ++bw)
      for (int mh = 0; mh < kMerge; ++mh)
        for (int mw = 0; mw < kMerge; ++mw) f(i++, bh * kMerge + mh, bw * kMerge + mw);
}

struct HostTensor {
  DType dtype = DType::kBF16;
  std::vector<std::int64_t> shape;
  std::vector<std::byte> bytes;
};

struct Linear {
  std::string name;
  std::int64_t n = 0, k = 0;
  int group = 0;      // ConvRot group; 0 for a bf16 linear
  bool int8 = false;  // false: bf16 weight
};

// The tower's tensors are int8, bf16, f32 or the u8 quantization configs; anything else is refused.
DType tensor_dtype(weights::DType d, const std::string& name) {
  switch (d) {
    case weights::DType::kI8: return DType::kI8;
    case weights::DType::kU8: return DType::kU8;
    case weights::DType::kBF16: return DType::kBF16;
    case weights::DType::kF32: return DType::kF32;
    default: fail(name + " has an unsupported dtype");
  }
}

int convrot_group(const std::vector<std::byte>& json) {
  const std::string s(reinterpret_cast<const char*>(json.data()), json.size());
  if (s.find("\"convrot\": true") == std::string::npos && s.find("\"convrot\":true") == std::string::npos) return 0;
  const auto at = s.find("convrot_groupsize");
  if (at == std::string::npos) return 0;
  const auto colon = s.find(':', at);
  return std::stoi(s.substr(colon + 1));
}

}  // namespace

std::pair<int, int> fit_size(int width, int height, std::int64_t max_pixels) {
  if (width <= 0 || height <= 0 || max_pixels <= 0) throw std::invalid_argument("qwen35_vision::fit_size: bad size");
  const double scale = std::min(1.0, std::sqrt(static_cast<double>(max_pixels) / (static_cast<double>(width) * height)));
  auto snap = [](double v) { return std::max(kGrid, static_cast<int>(std::lround(v / kGrid)) * kGrid); };
  int w = snap(width * scale), h = snap(height * scale);
  while (static_cast<std::int64_t>(w) * h > max_pixels && std::max(w, h) > kGrid) {
    if (h >= w) h -= kGrid;
    else w -= kGrid;
  }
  return {w, h};
}

std::uint64_t images_hash(const std::vector<Prepared>& images, std::uint64_t seed) {
  std::uint64_t h = seed;
  for (const Prepared& im : images) {
    const std::int32_t grid[2] = {im.grid_h, im.grid_w};
    h = xxh64(grid, sizeof grid, h);
    h = xxh64(im.patches.data(), im.patches.size() * sizeof(std::uint16_t), h);
  }
  return h;
}

Prepared prepare(const pipeline::Image& rgb, std::int64_t max_pixels, float weight) {
  if (rgb.channels != 3) throw std::invalid_argument("qwen35_vision::prepare: expected an RGB image");
  const auto [w, h] = fit_size(rgb.width, rgb.height, max_pixels);
  const pipeline::Image img = (w == rgb.width && h == rgb.height) ? rgb : pipeline::resize_lanczos(rgb, w, h);
  Prepared p;
  p.grid_h = h / kPatch;
  p.grid_w = w / kPatch;
  p.patches.resize(static_cast<std::size_t>(p.grid_h) * p.grid_w * kPatchInput);
  for_each_block_major(p.grid_h, p.grid_w, [&](std::int64_t i, int r, int c) {
    std::uint16_t* row = p.patches.data() + i * kPatchInput;
    std::size_t o = 0;
    for (int ch = 0; ch < 3; ++ch)
      for (int t = 0; t < kTemporal; ++t)
        for (int py = 0; py < kPatch; ++py)
          for (int px = 0; px < kPatch; ++px) {
            const std::size_t at = (static_cast<std::size_t>(r * kPatch + py) * img.width + (c * kPatch + px)) * 3 + ch;
            float x = static_cast<float>(img.pixels[at]) / 255.0f;
            if (weight != 1.0f) x = std::clamp(0.5f + weight * (x - 0.5f), 0.0f, 1.0f);
            row[o++] = to_bf16((x - 0.5f) / 0.5f);
          }
  });
  return p;
}

void rotary_tables(int grid_h, int grid_w, std::vector<float>& cos, std::vector<float>& sin) {
  constexpr int kHeadDim = kernels::vision::kHeadDim;  // 72
  constexpr int kRotDim = kHeadDim / 2;                // Qwen3_5VisionRotaryEmbedding(36)
  constexpr int kFreqs = kRotDim / 2;                  // 18 per axis
  float inv_freq[kFreqs];
  for (int i = 0; i < kFreqs; ++i) {
    inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(kRotDim));
  }
  const std::int64_t n = static_cast<std::int64_t>(grid_h) * grid_w;
  cos.assign(static_cast<std::size_t>(n * kHeadDim), 0.0f);
  sin.assign(static_cast<std::size_t>(n * kHeadDim), 0.0f);
  for_each_block_major(grid_h, grid_w, [&](std::int64_t i, int r, int c) {
    float* cr = cos.data() + i * kHeadDim;
    float* sr = sin.data() + i * kHeadDim;
    for (int f = 0; f < kFreqs; ++f) {
      const float ph = static_cast<float>(r) * inv_freq[f];
      const float pw = static_cast<float>(c) * inv_freq[f];
      // rotary_pos_emb = [h * inv_freq | w * inv_freq] (36), emb = cat(rotary_pos_emb, rotary_pos_emb) (72).
      for (int rep = 0; rep < 2; ++rep) {
        cr[rep * kRotDim + f] = std::cos(ph);
        sr[rep * kRotDim + f] = std::sin(ph);
        cr[rep * kRotDim + kFreqs + f] = std::cos(pw);
        sr[rep * kRotDim + kFreqs + f] = std::sin(pw);
      }
    }
  });
}

std::vector<std::vector<std::int32_t>> mrope_positions(std::int64_t length, const std::vector<std::int64_t>& image_starts,
                                                       const std::vector<std::pair<int, int>>& merged_grids) {
  if (image_starts.size() != merged_grids.size()) throw std::invalid_argument("qwen35_vision::mrope_positions: sizes differ");
  std::vector<std::vector<std::int32_t>> pos(3, std::vector<std::int32_t>(static_cast<std::size_t>(length)));
  std::int64_t i = 0;
  std::int32_t next = 0;
  for (std::size_t img = 0; img <= image_starts.size(); ++img) {
    const std::int64_t text_end = img < image_starts.size() ? image_starts[img] : length;
    if (text_end < i || text_end > length) throw std::invalid_argument("qwen35_vision::mrope_positions: image out of order");
    for (; i < text_end; ++i, ++next) {
      for (auto& axis : pos) axis[static_cast<std::size_t>(i)] = next;
    }
    if (img == image_starts.size()) break;
    const auto [gh, gw] = merged_grids[img];
    const std::int64_t count = static_cast<std::int64_t>(gh) * gw;
    if (i + count > length) throw std::invalid_argument("qwen35_vision::mrope_positions: image past the sequence end");
    for (int r = 0; r < gh; ++r) {
      for (int c = 0; c < gw; ++c, ++i) {
        pos[0][static_cast<std::size_t>(i)] = next;
        pos[1][static_cast<std::size_t>(i)] = next + r;
        pos[2][static_cast<std::size_t>(i)] = next + c;
      }
    }
    next += std::max(gh, gw);
  }
  return pos;
}

VisionPrompt build_prompt(const tok::Tokenizer& tokenizer, std::string_view prompt, const std::vector<Prepared>& images,
                          std::size_t max_length) {
  VisionPrompt p;
  std::vector<std::int32_t> text = tok::encode_moss_prompt(tokenizer, prompt, std::numeric_limits<std::size_t>::max());
  const std::size_t head = tokenizer.encode("<|im_start|>user\n").size();
  const auto special = [&](const char* s) {
    const auto ids = tokenizer.encode(s);
    if (ids.size() != 1) throw std::runtime_error(std::string("qwen35_vision: the tokenizer has no ") + s + " token");
    return ids[0];
  };
  const std::int32_t start = special("<|vision_start|>"), pad = special("<|image_pad|>"), end = special("<|vision_end|>");
  p.ids.assign(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(head));
  for (const Prepared& im : images) {
    p.ids.push_back(start);
    p.starts.push_back(static_cast<std::int64_t>(p.ids.size()));
    p.merged_grids.emplace_back(im.grid_h / kMerge, im.grid_w / kMerge);
    p.ids.insert(p.ids.end(), static_cast<std::size_t>(im.tokens()), pad);
    p.ids.push_back(end);
  }
  p.ids.insert(p.ids.end(), text.begin() + static_cast<std::ptrdiff_t>(head), text.end());
  if (p.ids.size() > max_length) {
    throw std::invalid_argument("qwen35_vision: the prompt with its reference images is " + std::to_string(p.ids.size()) +
                                " tokens, over the limit of " + std::to_string(max_length) +
                                "; use smaller images (max_pixels) or a shorter prompt");
  }
  const auto pos = mrope_positions(static_cast<std::int64_t>(p.ids.size()), p.starts, p.merged_grids);
  for (const auto& axis : pos) p.positions.insert(p.positions.end(), axis.begin(), axis.end());
  return p;
}

struct VisionEncoder::Impl {
  std::map<std::string, HostTensor, std::less<>> tensors;  // canonical names: "blocks.3.attn.qkv.weight", "merger.norm.bias"
  std::map<std::string, Linear, std::less<>> linears;
  std::vector<std::uint16_t> pos_table;  // bf16 [2304, 1152]
  std::unique_ptr<kernels::Gemm> gemm;

  const HostTensor& at(const std::string& name) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) fail("the text encoder file has no vision tensor " + name);
    return it->second;
  }
};

namespace {

const char* kPrefixes[] = {"model.visual.", "visual."};

std::string canonical(const std::string& name) {
  for (const char* p : kPrefixes) {
    if (name.rfind(p, 0) == 0) return name.substr(std::strlen(p));
  }
  return {};
}

}  // namespace

bool VisionEncoder::available(const std::filesystem::path& te_file) {
  try {
    weights::SafetensorsFile f(te_file);
    for (const auto& t : f.tensors()) {
      if (!canonical(t.name).empty()) return true;
    }
  } catch (const std::exception&) {
  }
  return false;
}

VisionEncoder::VisionEncoder(const std::filesystem::path& te_file) : impl_(std::make_unique<Impl>()) {
  weights::SafetensorsFile f(te_file);
  std::ifstream in(te_file, std::ios::binary);
  if (!in) fail("cannot open " + te_file.string());
  for (const auto& t : f.tensors()) {
    const std::string name = canonical(t.name);
    if (name.empty()) continue;
    HostTensor h;
    h.dtype = tensor_dtype(t.dtype, t.name);
    h.shape = t.shape;
    h.bytes.resize(t.nbytes);
    in.seekg(static_cast<std::streamoff>(t.offset));
    in.read(reinterpret_cast<char*>(h.bytes.data()), static_cast<std::streamsize>(t.nbytes));
    if (!in) fail("cannot read " + t.name);
    impl_->tensors.emplace(name, std::move(h));
  }
  if (impl_->tensors.empty()) fail("the text encoder file has no vision tower (model.visual.*)");

  auto add_linear = [&](const std::string& module, std::int64_t n, std::int64_t k) {
    const HostTensor& w = impl_->at(module + ".weight");
    Linear l{module, n, k, 0, w.dtype == DType::kI8};
    if (w.shape != std::vector<std::int64_t>{n, k}) fail(module + ".weight has an unexpected shape");
    if (l.int8) {
      const HostTensor& s = impl_->at(module + ".weight_scale");
      if (s.dtype != DType::kF32 || s.bytes.size() != static_cast<std::size_t>(n) * 4) fail(module + ".weight_scale must be f32 [n]");
      const auto q = impl_->tensors.find(module + ".comfy_quant");
      if (q != impl_->tensors.end()) l.group = convrot_group(q->second.bytes);
    } else if (w.dtype != DType::kBF16) {
      fail(module + ".weight must be int8 or bf16");
    }
    const HostTensor& b = impl_->at(module + ".bias");
    if (b.dtype != DType::kBF16 || b.bytes.size() != static_cast<std::size_t>(n) * 2) fail(module + ".bias must be bf16 [n]");
    impl_->linears.emplace(module, l);
  };
  for (int i = 0; i < kDepth; ++i) {
    const std::string p = "blocks." + std::to_string(i) + ".";
    add_linear(p + "attn.qkv", 3 * kDim, kDim);
    add_linear(p + "attn.proj", kDim, kDim);
    add_linear(p + "mlp.linear_fc1", kMlp, kDim);
    add_linear(p + "mlp.linear_fc2", kDim, kMlp);
    for (const char* n : {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"}) {
      if (impl_->at(p + n).dtype != DType::kBF16) fail(p + n + " must be bf16");
    }
  }
  add_linear("merger.linear_fc1", kDim * 4, kDim * 4);
  add_linear("merger.linear_fc2", kOut, kDim * 4);
  const HostTensor& patch = impl_->at("patch_embed.proj.weight");
  if (patch.dtype != DType::kBF16 || patch.bytes.size() != static_cast<std::size_t>(kDim) * kPatchInput * 2) {
    fail("patch_embed.proj.weight must be bf16 [1152, 3, 2, 16, 16]");
  }
  const HostTensor& pos = impl_->at("pos_embed.weight");
  if (pos.dtype != DType::kBF16 || pos.bytes.size() != static_cast<std::size_t>(kPosSide) * kPosSide * kDim * 2) {
    fail("pos_embed.weight must be bf16 [2304, 1152]");
  }
  impl_->pos_table.resize(pos.bytes.size() / 2);
  std::memcpy(impl_->pos_table.data(), pos.bytes.data(), pos.bytes.size());
}

VisionEncoder::~VisionEncoder() = default;

std::size_t VisionEncoder::workspace_bytes(std::int64_t patches) {
  // Weights as stored: the blocks' linears INT8 (+ f32 scales, bf16 biases), the rest bf16 (bf16 linears counted at
  // two bytes per weight bound an all-bf16 file too), each tensor aligned.
  const auto lin = [](std::size_t n, std::size_t k) { return n * k * 2 + n * 4 + n * 2 + 3 * runtime::kArenaAlign; };
  std::size_t weights = kDepth * (lin(3 * kDim, kDim) + lin(kDim, kDim) + lin(kMlp, kDim) + lin(kDim, kMlp) +
                                  4 * (kDim * 2 + runtime::kArenaAlign));
  weights += lin(kDim * 4, kDim * 4) + lin(kOut, kDim * 4) + 2 * (kDim * 2 + runtime::kArenaAlign);
  weights += static_cast<std::size_t>(kDim) * kPatchInput * 2 + kDim * 2 + 2 * runtime::kArenaAlign;
  const auto n = static_cast<std::size_t>(patches);
  // Every buffer encode() holds at once: the uploads (patches, positions, rotary tables), the block buffers, and the
  // merger's fc1 next to the output.
  const std::size_t act = n * (kPatchInput * 2 /*px*/ + kDim * 2 /*pos*/ + 72 * 4 * 2 /*cos, sin*/ + kDim * 2 /*x*/ +
                               kDim * 2 /*h*/ + kDim * 2 /*y*/ + 3 * kDim * 2 /*qkv*/ + 3 * 16 * 128 * 2 /*q,k,v pad*/ +
                               16 * 128 * 2 /*attn*/ + kMlp * 2 /*mlp*/) +
                          n / 4 * (kDim * 4 * 2 /*fc1*/ + kOut * 2 /*out*/);
  const std::size_t one_linear = static_cast<std::size_t>(kDim) * 4 * kDim * 4 * 2;  // the largest dequantized weight
  return weights + act + one_linear + kernels::Gemm::kWorkspaceBytes +
         attention::flash_fwd_workspace_bytes(patches, kernels::vision::kHeads, static_cast<int>(patches)) +
         (std::size_t{64} << 10) * 64;
}

Tensor VisionEncoder::encode(const std::vector<Prepared>& images, runtime::DeviceArena& arena, cudaStream_t stream) {
  if (images.empty()) throw std::invalid_argument("qwen35_vision::encode: no images");
  Impl& m = *impl_;
  if (!m.gemm) m.gemm = std::make_unique<kernels::Gemm>();
  std::int64_t patches = 0, tokens = 0;
  std::vector<std::int32_t> cu{0};
  int max_len = 0;
  for (const auto& im : images) {
    const std::int64_t n = static_cast<std::int64_t>(im.grid_h) * im.grid_w;
    if (n == 0 || im.grid_h % kMerge || im.grid_w % kMerge) throw std::invalid_argument("qwen35_vision::encode: bad grid");
    patches += n;
    tokens += im.tokens();
    cu.push_back(static_cast<std::int32_t>(patches));
    max_len = std::max(max_len, static_cast<int>(n));
  }

  // Host side: patches, position embeddings (bilinear, fp32 sum rounded to bf16), rotary tables.
  std::vector<std::uint16_t> host_patches, host_pos;
  std::vector<float> host_cos, host_sin;
  host_patches.reserve(static_cast<std::size_t>(patches) * kPatchInput);
  host_pos.reserve(static_cast<std::size_t>(patches) * kDim);
  for (const auto& im : images) {
    host_patches.insert(host_patches.end(), im.patches.begin(), im.patches.end());
    const auto hg = linspace(0.0f, static_cast<float>(kPosSide - 1), im.grid_h);
    const auto wg = linspace(0.0f, static_cast<float>(kPosSide - 1), im.grid_w);
    for_each_block_major(im.grid_h, im.grid_w, [&](std::int64_t, int r, int c) {
      const int hf = static_cast<int>(hg[static_cast<std::size_t>(r)]), wf = static_cast<int>(wg[static_cast<std::size_t>(c)]);
      const int hc = std::min(hf + 1, kPosSide - 1), wc = std::min(wf + 1, kPosSide - 1);
      const float dh = hg[static_cast<std::size_t>(r)] - static_cast<float>(hf), dw = wg[static_cast<std::size_t>(c)] - static_cast<float>(wf);
      const int idx[4] = {hf * kPosSide + wf, hf * kPosSide + wc, hc * kPosSide + wf, hc * kPosSide + wc};
      const float wt[4] = {(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw};
      for (int d = 0; d < kDim; ++d) {
        float acc = 0.0f;
        for (int q = 0; q < 4; ++q) acc += from_bf16(m.pos_table[static_cast<std::size_t>(idx[q]) * kDim + d]) * wt[q];
        host_pos.push_back(to_bf16(acc));
      }
    });
    std::vector<float> c, s;
    rotary_tables(im.grid_h, im.grid_w, c, s);
    host_cos.insert(host_cos.end(), c.begin(), c.end());
    host_sin.insert(host_sin.end(), s.begin(), s.end());
  }

  const std::size_t mark = arena.mark();
  Tensor out = arena.alloc_tensor(DType::kBF16, {tokens, kOut});
  try {
    runtime::StackScope scratch(arena);
    auto upload = [&](const void* src, std::size_t bytes, DType dtype, std::initializer_list<std::int64_t> shape) {
      Tensor t = arena.alloc_tensor(dtype, shape);
      MI_CUDA_CHECK(cudaMemcpyAsync(t.data, src, bytes, cudaMemcpyHostToDevice, stream));
      return t;
    };
    std::map<std::string, Tensor, std::less<>> dev;
    for (const auto& [name, t] : m.tensors) {
      if (name.ends_with(".comfy_quant") || name == "pos_embed.weight") continue;
      Tensor d = arena.alloc_tensor(t.dtype, {static_cast<std::int64_t>(t.bytes.size() / dtype_size(t.dtype))});
      MI_CUDA_CHECK(cudaMemcpyAsync(d.data, t.bytes.data(), t.bytes.size(), cudaMemcpyHostToDevice, stream));
      dev.emplace(name, d);
    }
    Tensor dq = arena.alloc_tensor(DType::kBF16, {static_cast<std::int64_t>(kDim) * 4 * kDim * 4});
    void* gemm_ws = arena.alloc_stack(kernels::Gemm::kWorkspaceBytes);

    auto linear = [&](const std::string& module, const Tensor& x, const Tensor& y) {
      const Linear& l = m.linears.at(module);
      Tensor w;
      if (l.int8) {
        w = Tensor::view(dq.data, DType::kBF16, Device::kCUDA, {l.n, l.k});
        kernels::vision::dequantize_convrot(dev.at(module + ".weight").reshaped({l.n, l.k}), dev.at(module + ".weight_scale"),
                                            l.group, w, stream);
      } else {
        w = dev.at(module + ".weight").reshaped({l.n, l.k});
      }
      const Tensor bias = dev.at(module + ".bias");
      m.gemm->linear(y, x, w, &bias, gemm_ws, kernels::Gemm::kWorkspaceBytes, stream);
    };

    Tensor px = upload(host_patches.data(), host_patches.size() * 2, DType::kBF16, {patches, kPatchInput});
    Tensor pos = upload(host_pos.data(), host_pos.size() * 2, DType::kBF16, {patches, kDim});
    Tensor cos = upload(host_cos.data(), host_cos.size() * 4, DType::kF32, {patches, 72});
    Tensor sin = upload(host_sin.data(), host_sin.size() * 4, DType::kF32, {patches, 72});
    Tensor cu_dev = upload(cu.data(), cu.size() * 4, DType::kI32, {static_cast<std::int64_t>(cu.size())});

    Tensor x = arena.alloc_tensor(DType::kBF16, {patches, kDim});
    Tensor h = arena.alloc_tensor(DType::kBF16, {patches, kDim});
    Tensor y = arena.alloc_tensor(DType::kBF16, {patches, kDim});
    Tensor qkv = arena.alloc_tensor(DType::kBF16, {patches, 3 * kDim});
    Tensor qp = arena.alloc_tensor(DType::kBF16, {patches, 16, 128});
    Tensor kp = arena.alloc_tensor(DType::kBF16, {patches, 16, 128});
    Tensor vp = arena.alloc_tensor(DType::kBF16, {patches, 16, 128});
    Tensor attn = arena.alloc_tensor(DType::kBF16, {patches, 16, 128});
    Tensor mlp = arena.alloc_tensor(DType::kBF16, {patches, kMlp});
    const std::size_t fa_bytes =
        attention::flash_fwd_workspace_bytes(patches, kernels::vision::kHeads, static_cast<int>(patches));
    void* fa_ws = arena.alloc_stack(fa_bytes);

    // Patch embedding: Conv3d with stride == kernel is a linear over the flattened (C, T, P, P) patch.
    const Tensor patch_w = dev.at("patch_embed.proj.weight").reshaped({kDim, kPatchInput});
    const Tensor patch_b = dev.at("patch_embed.proj.bias");
    m.gemm->linear(x, px, patch_w, &patch_b, gemm_ws, kernels::Gemm::kWorkspaceBytes, stream);
    kernels::residual_add(x, pos, x, stream);

    const float scale = 1.0f / std::sqrt(static_cast<float>(kernels::vision::kHeadDim));
    for (int i = 0; i < kDepth; ++i) {
      const std::string p = "blocks." + std::to_string(i) + ".";
      kernels::vision::layer_norm(x, dev.at(p + "norm1.weight"), dev.at(p + "norm1.bias"), 1e-6f, h, stream);
      linear(p + "attn.qkv", h, qkv);
      kernels::vision::qkv_rope_pad(qkv, cos, sin, qp, kp, vp, stream);
      attention::flash_fwd(qp, kp, vp, attn, cu_dev, cu_dev, max_len, /*causal=*/false, scale, fa_ws, fa_bytes, stream);
      kernels::vision::unpad_heads(attn, h, stream);
      linear(p + "attn.proj", h, y);
      kernels::residual_add(x, y, x, stream);
      kernels::vision::layer_norm(x, dev.at(p + "norm2.weight"), dev.at(p + "norm2.bias"), 1e-6f, h, stream);
      linear(p + "mlp.linear_fc1", h, mlp);
      kernels::vision::gelu_tanh(mlp, stream);
      linear(p + "mlp.linear_fc2", mlp, y);
      kernels::residual_add(x, y, x, stream);
    }

    // Merger: LayerNorm per patch, then each 2x2 block (consecutive in block order) as one 4608-wide row.
    kernels::vision::layer_norm(x, dev.at("merger.norm.weight"), dev.at("merger.norm.bias"), 1e-6f, h, stream);
    const Tensor merged = h.reshaped({tokens, kDim * 4});
    Tensor fc1 = arena.alloc_tensor(DType::kBF16, {tokens, kDim * 4});
    linear("merger.linear_fc1", merged, fc1);
    kernels::vision::gelu_erf(fc1, stream);
    linear("merger.linear_fc2", fc1, out);
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (...) {
    arena.rollback(mark);
    throw;
  }
  return out;
}

}  // namespace mi::qwen35_vision
