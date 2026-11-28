// Runtime LoRA support for the HIP backend. Only the DiT block GEMMs matter here: each adapter is a low-rank pair
// A [r, k], B [n, r] and the engine's delta is coef · B · (A x), with coef = strength · alpha / rank. The factors are
// rounded to bf16 on load and the coef is folded into B's columns (a rank column's coef scales the whole output column
// contribution), so the device only sees two bf16 GEMMs per site: t = x · Aᵀ and delta = t · Bᵀ. The delta is added to
// the base GEMM's bf16 output before the site's epilogue (qk-norm, gated residual or SiTU-GLU).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "hip_block.h"
#include "runtime.h"

namespace mi::hip {

using bf16 = hip_block::bf16_bits;

enum class LoraSite : int { kQkv = 0, kO = 1, kUpGate = 2, kDown = 3 };
inline constexpr int kLoraSites = 4;

struct LoraBranch {
  bool active = false;
  int rank = 0, n = 0, k = 0;
  const bf16* a = nullptr;  // [rank, k] row-major
  const bf16* b = nullptr;  // [n, rank] row-major, coef folded into columns
};

struct LoraBlock {
  LoraBranch site[kLoraSites];
};

class Lora {
 public:
  // Throws std::runtime_error on an unreadable file or an adapter with no supported target. `blocks` is the DiT's
  // layer count; `strength` scales every adapter in the file.
  Lora(const std::filesystem::path& file, float strength, int blocks);

  bool loaded() const { return loaded_; }
  const LoraBlock& block(int l) const { return blocks_[static_cast<std::size_t>(l)]; }
  int max_rank() const { return max_rank_; }
  int max_n() const { return max_n_; }
  int adapters() const { return adapters_; }
  std::size_t device_bytes() const { return weights_.bytes(); }

  struct Loaded {
    std::vector<bf16> a;  // [rank, k]
    std::vector<bf16> b;  // [n, rank]
    int rank = 0, n = 0, k = 0;
  };

 private:
  std::string path_;
  bool loaded_ = false;
  int adapters_ = 0, max_rank_ = 0, max_n_ = 0;
  DeviceBuffer weights_;  // all blocks' A/B factors, contiguous
  std::vector<LoraBlock> blocks_;
};

}  // namespace mi::hip
