#include "lora.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "half.h"
#include "safetensors.h"
#include "weights/quant/quant_codec.h"

namespace mi::hip {
namespace {

using bf16 = hip_block::bf16_bits;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("hip lora: " + what); }

// Bring kohya's underscored names into the common dotted form:
// lora_unet_layers_0_proj_ffn_up_gate_lora_down_weight -> layers.0.proj.ffn.up.gate.lora.down.weight.
std::string normalize(std::string name) {
  static const char* kPrefixes[] = {"model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.",
                                    "lora_unet_", "lora_transformer_", "lycoris_"};
  for (const char* p : kPrefixes) {
    const std::size_t n = std::strlen(p);
    if (name.compare(0, n, p) == 0) {
      name.erase(0, n);
      break;
    }
  }
  for (char& c : name) {
    if (c == '_') c = '.';
  }
  std::string out;
  for (const char c : name) {
    if (c != '.' || out.empty() || out.back() != '.') out += c;
  }
  return out;
}

struct Target {
  int layer = -1;
  int site = -1;
  std::string module;
  std::string alpha_name;
  const st::TensorInfo* down = nullptr;
  const st::TensorInfo* up = nullptr;
};

bool parse_target(const std::string& key, const st::File& file, Target& t) {
  const std::string name = normalize(key);
  constexpr std::string_view kLayers = "layers.";
  if (name.compare(0, kLayers.size(), kLayers) != 0) return false;
  std::size_t at = kLayers.size(), digits = 0;
  while (at < name.size() && name[at] >= '0' && name[at] <= '9') ++at, ++digits;
  if (digits == 0 || at >= name.size() || name[at] != '.') return false;
  t.layer = std::stoi(name.substr(kLayers.size(), digits));
  ++at;
  if (name.compare(at, 5, "proj.") != 0) return false;
  at += 5;
  const std::string rest = name.substr(at);
  // Longest module names first.
  static const std::pair<std::string_view, int> kModules[] = {
      {"ffn.up.gate.", static_cast<int>(LoraSite::kUpGate)}, {"ffn.down.", static_cast<int>(LoraSite::kDown)},
      {"qkv.", static_cast<int>(LoraSite::kQkv)}, {"q.", static_cast<int>(LoraSite::kQkv)},
      {"k.", static_cast<int>(LoraSite::kQkv)}, {"v.", static_cast<int>(LoraSite::kQkv)},
      {"o.", static_cast<int>(LoraSite::kO)}};
  for (const auto& [m, site] : kModules) {
    if (rest.compare(0, m.size(), m) != 0) continue;
    t.site = site;
    t.module = std::string(m.substr(0, m.size() - 1));
    t.alpha_name = "layers." + std::to_string(t.layer) + ".proj." + t.module + ".alpha";
    const std::string suffix = rest.substr(m.size());
    // `name` has already been normalized (underscores became dots), so both spellings must be compared in their
    // normalized form: lora_down.weight -> lora.down.weight, lora_A.weight -> lora.A.weight.
    const bool down = suffix == "lora.down.weight" || suffix == "lora.A.weight" || suffix == "lora.A";
    const bool up = suffix == "lora.up.weight" || suffix == "lora.B.weight" || suffix == "lora.B";
    if (!down && !up) return false;
    const st::TensorInfo* info = file.find(key);
    if (down) t.down = info;
    if (up) t.up = info;
    return true;
  }
  return false;
}

// LoRA factors ship as F32, BF16 or F16; widen them to fp32 before rounding to bf16 for the device.
std::vector<float> read_f32(const st::File& file, const st::TensorInfo& t) {
  if (t.dtype == "F32") {
    std::vector<float> v(static_cast<std::size_t>(t.bytes / 4));
    file.read(t, v.data());
    return v;
  }
  if (t.dtype == "BF16" || t.dtype == "F16") {
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(t.bytes / 2));
    file.read(t, raw.data());
    std::vector<float> v(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
      v[i] = t.dtype == "BF16" ? mi::quant::bf16_to_float(raw[i]) : half_to_float(raw[i]);
    return v;
  }
  fail(t.name + " is " + t.dtype + ", expected F32, BF16 or F16");
}

// One adapter's factors as bf16, with the coefficient folded into B's columns.
Lora::Loaded read_one(const st::File& file, const st::TensorInfo& down, const st::TensorInfo& up, float coef, int n,
                      int k) {
  const std::vector<float> d = read_f32(file, down);
  const std::vector<float> u = read_f32(file, up);
  if (down.shape.size() != 2 || up.shape.size() != 2) fail("low-rank factors must be matrices");
  const int rank = static_cast<int>(down.shape[0]);
  if (rank <= 0 || down.shape[1] != k || up.shape[0] != n || up.shape[1] != rank) {
    fail("a LoRA factor has an unexpected shape for " + down.name);
  }
  if (!std::isfinite(coef)) fail("a LoRA coefficient is not finite");
  Lora::Loaded out;
  out.rank = rank;
  out.n = n;
  out.k = k;
  out.a.resize(static_cast<std::size_t>(rank) * k);
  out.b.resize(static_cast<std::size_t>(n) * rank, 0);
  for (std::size_t i = 0; i < out.a.size(); ++i) out.a[i] = mi::quant::float_to_bf16(d[i]);
  for (int row = 0; row < n; ++row) {
    for (int c = 0; c < rank; ++c) {
      out.b[static_cast<std::size_t>(row) * rank + c] =
          mi::quant::float_to_bf16(u[static_cast<std::size_t>(row) * rank + c] * coef);
    }
  }
  return out;
}

struct Adapter {
  int layer = 0, site = 0, rank = 0;
  std::string module;
  Lora::Loaded data;
};

}  // namespace

Lora::Lora(const std::filesystem::path& file, float strength, int blocks) {
  const std::u8string path_u8 = file.u8string();
  path_ = std::string(path_u8.begin(), path_u8.end());
  blocks_.assign(static_cast<std::size_t>(blocks), LoraBlock{});
  if (file.empty()) return;
  if (!std::filesystem::is_regular_file(file)) fail("not a file: " + path_);

  st::File st_file(file);
  // Alpha names are built in normalized form, but safetensors lookup is by the file's original key. Keep the mapping
  // so `lora_unet_layers_0_proj_q.alpha` or `diffusion_model.layers.0.proj.q.alpha` is found from
  // `layers.0.proj.q.alpha`.
  std::unordered_map<std::string, std::string> original_key;
  for (const st::TensorInfo& info : st_file.tensors()) original_key.emplace(normalize(info.name), info.name);
  // Group the file's keys by (layer, module). A module may have lora_down/lora_up/alpha separately.
  std::vector<Target> targets;
  for (const st::TensorInfo& info : st_file.tensors()) {
    Target t;
    if (!parse_target(info.name, st_file, t)) continue;
    if (t.layer < 0 || t.layer >= blocks) continue;
    std::size_t slot = targets.size();
    for (std::size_t i = 0; i < targets.size(); ++i) {
      if (targets[i].layer == t.layer && targets[i].module == t.module) {
        slot = i;
        break;
      }
    }
    if (slot == targets.size()) targets.push_back(t);
    Target& found = targets[slot];
    if (t.down) found.down = t.down;
    if (t.up) found.up = t.up;
  }

  constexpr int kDim = 3072, kFfn = 8192;
  // PEFT/diffusers files put alpha in lora_adapter_metadata instead of an alpha tensor. Without it the adapter runs
  // at alpha/rank = 1, which is wrong for those files (the engine uses the same fallback).
  const std::optional<double> meta_alpha = st_file.lora_metadata_alpha();
  std::vector<Adapter> adapters;
  for (const Target& t : targets) {
    if (t.down == nullptr || t.up == nullptr) continue;
    const int rank = static_cast<int>(t.down->shape[0]);
    if (rank <= 0) fail("a LoRA rank is zero");
    float alpha = 0;  // 0: no alpha anywhere, use alpha / rank = 1
    const auto ak = original_key.find(t.alpha_name);
    if (ak != original_key.end()) {
      const st::TensorInfo* ai = st_file.find(ak->second);
      if (ai != nullptr) {
        const std::vector<float> av = read_f32(st_file, *ai);
        if (!av.empty() && std::isfinite(av[0])) alpha = av[0];
      }
    }
    if (alpha == 0 && meta_alpha.has_value() && std::isfinite(*meta_alpha) && *meta_alpha > 0)
      alpha = static_cast<float>(*meta_alpha);
    if (alpha == 0) alpha = static_cast<float>(rank);
    const float coef = strength * (alpha / static_cast<float>(rank));
    int n = 0, k = 0;
    switch (static_cast<LoraSite>(t.site)) {
      // A fused adapter stores q|k|v as one [3 kDim, rank] up factor; the engine splits it into three sites. The branch
      // assembly below keeps a fused factor intact (n == 3 kDim) and concatenates separate q/k/v factors into one.
      case LoraSite::kQkv: n = t.module == "qkv" ? 3 * kDim : kDim, k = kDim; break;
      case LoraSite::kO: n = kDim, k = kDim; break;
      case LoraSite::kUpGate: n = 2 * kFfn, k = kDim; break;
      case LoraSite::kDown: n = kDim, k = kFfn; break;
    }
    Adapter a;
    a.layer = t.layer;
    a.site = t.site;
    a.module = t.module;
    a.data = read_one(st_file, *t.down, *t.up, coef, n, k);
    adapters.push_back(std::move(a));
    ++adapters_;
  }
  if (adapters.empty()) fail("no DiT block adapters found in " + path_);

  // Assemble one branch per (block, site). q, k and v adapters share the fused qkv site's output rows.
  struct Branch {
    int layer = 0, site = 0;
    Loaded data;
  };
  std::vector<Branch> branches;
  for (int l = 0; l < blocks; ++l) {
    for (int s = 0; s < kLoraSites; ++s) {
      std::vector<std::size_t> parts;
      for (std::size_t i = 0; i < adapters.size(); ++i) {
        if (adapters[i].layer == l && adapters[i].site == s) parts.push_back(i);
      }
      if (parts.empty()) continue;
      Branch br;
      br.layer = l;
      br.site = s;
      if (s == static_cast<int>(LoraSite::kQkv) && adapters[parts[0]].data.n != 3 * kDim) {
        int rank = 0;
        for (std::size_t i : parts) rank += adapters[i].data.rank;
        br.data.rank = rank;
        br.data.n = 3 * kDim;
        br.data.k = kDim;
        br.data.a.assign(static_cast<std::size_t>(rank) * kDim, 0);
        br.data.b.assign(static_cast<std::size_t>(3 * kDim) * rank, 0);
        int col = 0;
        for (std::size_t i : parts) {
          const Adapter& part = adapters[i];
          const int r = part.data.rank;
          int row0 = 0;
          if (part.module == "k") row0 = kDim;
          if (part.module == "v") row0 = 2 * kDim;
          for (int j = 0; j < r; ++j) {
            std::memcpy(br.data.a.data() + static_cast<std::size_t>(col + j) * kDim,
                        part.data.a.data() + static_cast<std::size_t>(j) * kDim,
                        static_cast<std::size_t>(kDim) * sizeof(bf16));
          }
          for (int row = 0; row < kDim; ++row) {
            std::memcpy(br.data.b.data() + static_cast<std::size_t>(row0 + row) * rank + col,
                        part.data.b.data() + static_cast<std::size_t>(row) * r,
                        static_cast<std::size_t>(r) * sizeof(bf16));
          }
          col += r;
        }
      } else {
        br.data = std::move(adapters[parts[0]].data);
      }
      branches.push_back(std::move(br));
    }
  }

  std::uint64_t total = 0;
  for (const Branch& br : branches) {
    total += (static_cast<std::uint64_t>(br.data.rank) * br.data.k + static_cast<std::uint64_t>(br.data.n) * br.data.rank) * 2;
    max_rank_ = std::max(max_rank_, br.data.rank);
    max_n_ = std::max(max_n_, br.data.n);
  }
  HostBuffer host(total);
  auto* base = static_cast<std::byte*>(host.data());
  std::uint64_t off = 0;
  for (const Branch& br : branches) {
    const std::size_t a_bytes = static_cast<std::size_t>(br.data.rank) * br.data.k * 2;
    const std::size_t b_bytes = static_cast<std::size_t>(br.data.n) * br.data.rank * 2;
    std::memcpy(base + off, br.data.a.data(), a_bytes);
    const bf16* a_dev = reinterpret_cast<const bf16*>(static_cast<std::byte*>(nullptr) + off);
    off += a_bytes;
    std::memcpy(base + off, br.data.b.data(), b_bytes);
    const bf16* b_dev = reinterpret_cast<const bf16*>(static_cast<std::byte*>(nullptr) + off);
    off += b_bytes;
    LoraBranch& site = blocks_[static_cast<std::size_t>(br.layer)].site[br.site];
    site.active = true;
    site.rank = br.data.rank;
    site.n = br.data.n;
    site.k = br.data.k;
    site.a = a_dev;
    site.b = b_dev;
  }
  weights_ = DeviceBuffer(total);
  weights_.upload(host.data(), total);
  auto* dev = static_cast<std::byte*>(weights_.data());
  for (LoraBlock& block : blocks_) {
    for (LoraBranch& site : block.site) {
      if (!site.active) continue;
      site.a = reinterpret_cast<const bf16*>(dev + reinterpret_cast<std::uintptr_t>(site.a));
      site.b = reinterpret_cast<const bf16*>(dev + reinterpret_cast<std::uintptr_t>(site.b));
    }
  }
  loaded_ = true;
}

}  // namespace mi::hip
