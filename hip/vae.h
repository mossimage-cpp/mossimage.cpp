// The FLUX.2 VAE decoder as the MOSS-Image pipeline runs it (the engine's src/models/flux2_vae, pipeline_moss_image.py
// 430-438): the DiT's final latents, fp32 NCHW [128, h, w], to RGB8 [16h, 16w, 3]. Activations are fp32 NHWC, as in the
// engine's reference precision (DecodeMode::reference()):
//   latent de-normalization + unpack -> post_quant_conv 1x1 -> conv_in 3x3 -> mid resnet 0 -> mid attention
//   -> mid resnet 1 -> 4 up blocks (3 resnets each, nearest 2x upsample + 3x3 conv after the first three)
//   -> GroupNorm + SiLU -> conv_out 3x3 -> clamp and the image processor's rounding to uint8.
// Convolutions accumulate in fp32. Operands and activation buffers follow the device: gfx103x (RDNA2, no bf16
// instructions) rounds each operand to fp16 (one v_dot2_f32_f16 per MAC), gfx11/gfx12 rounds to bf16 (one native WMMA
// bf16 product per MAC), and both keep the activation buffers in the same 16-bit format. MI_HIP_VAE_FP16=0 /
// MI_HIP_VAE_BF16=0 restore the exact TF32 product (four bf16-piece dot products); MI_HIP_VAE_FP16_ACT=0 /
// MI_HIP_VAE_BF16_ACT=0 restore fp32 activations. Other devices keep the TF32/fp32 defaults. The attention's linears,
// scores and P V follow the operand mode; GroupNorm statistics, the conv_out result and the RGB8 conversion stay fp32.
// The nearest upsample is read inside the following convolution. Accumulation orders differ from cuDNN's and torch's,
// so outputs are close to the engine's, not bit-identical (bf16 differs by at most 8 RGB levels).
//
// Memory (untiled): two regions of 2c x 16h x 16w elements each (c = 96 for the small decoder; 16-bit on gfx103x and
// gfx11/gfx12, fp32 elsewhere) as the engine's ping-pong schedule, the latents, the fp32 conv_out result, the RGB output
// and the GroupNorm accumulators; the mid attention's scratch lives in the second region. workspace_bytes() is the total;
// decode() refuses to start when the device's free memory cannot hold it.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"
#include "vae_kernels.h"

namespace mi::hip {

struct VaeDecodeStats {
  double seconds = 0;              // decode() wall time, the latents' upload and the RGB download included
  std::uint64_t device_bytes = 0;  // device memory decode() allocated (weights not included)
  // With set_profile(true): device milliseconds per op kind, in order of first use.
  std::vector<std::pair<std::string, double>> profile;
};

// Exact band-tiled schedule (the engine's plan step 23). The tail is 17 units in execution order: mid resnet 1 (0),
// up_blocks[b]'s three resnets and upsampler (1-4, 5-8, 9-12, 13-15), and conv_norm_out + conv_out + RGB (16). A
// segment runs units [first_unit, end_unit) from the checkpoint before first_unit; its bands are full-width row
// bands of `band_rows` rows at that segment's *output* resolution (all rows: one band). All the bands tile the
// image exactly, including a shorter final band. The head (latent unpack through mid attention) always runs untiled.
struct VaeTilePlan {
  struct Segment {
    int first_unit = 0, end_unit = 0;
    std::uint32_t band_rows = 0;  // band height at the segment's output resolution
  };
  std::uint32_t height = 0, width = 0;  // latent grid the plan is for (image = 16h x 16w)
  int base_channels = 0;               // decoder width the plan is for
  int tile = 0;                        // band height in image pixels (a multiple of 8)
  std::vector<Segment> segments;
  std::uint64_t workspace_bytes = 0;  // device bytes the planner estimated for the activation mode at plan time
  bool fp16_act = false;              // 16-bit vs fp32 activation storage the plan was made for
  double relative_cost = 0;           // estimated tail work (conv MACs + memory traffic) vs the untiled tail

  std::string describe() const;
};

class VaeDecoder {
 public:
  // Loads the decoder ("decoder.*", "post_quant_conv.*") and the latent statistics ("bn.running_mean", "bn.running_var")
  // from a FLUX.2 VAE safetensors file in diffusers' naming, fp32 or bf16. The widths come from the weights' shapes
  // (decoder_block_out_channels c, 2c, 4c, 4c; c a multiple of 32).
  VaeDecoder(Device& device, const std::filesystem::path& safetensors_path);
  ~VaeDecoder();
  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;

  int base_channels() const { return base_; }
  std::uint64_t weight_bytes() const;
  // Device bytes decode() allocates for an h x w latent grid, for a decoder of base width c (96: the MOSS bundle's).
  static std::uint64_t workspace_bytes(int base_channels, std::uint32_t h, std::uint32_t w);
  std::uint64_t workspace_bytes(std::uint32_t h, std::uint32_t w) const { return workspace_bytes(base_, h, w); }
  // Throws std::runtime_error naming both sizes when the device's free memory is below workspace_bytes(h, w).
  void require_fits(std::uint32_t h, std::uint32_t w) const;

  // The convolutions' products: kTf32 (exact TF32 products), kTf32Drop, kFp16 (the gfx103x default) or kBf16 (the
  // gfx11/gfx12 default). Must agree with the weights' packing and activation mode chosen at construction. Only the
  // operand-rounding family may change (e.g. kTf32 -> kTf32Drop), not fp16 <-> bf16.
  void set_conv_products(hip_vae::Products products);
  // The operand format actually used by this decoder's convolutions (the one above after any set_conv_products call).
  hip_vae::Products conv_products() const { return conv_products_; }
  // Times every op with device events (VaeDecodeStats::profile).
  void set_profile(bool on) { profile_ = on; }

  // latents: fp32 NCHW [128, h, w], the pipeline's final latents. rgb: resized to 16h x 16w x 3. image: when not null, the
  // decoder's fp32 output [16h, 16w, 3] before the clamp. Runs on its own stream and returns when the RGB is on the host.
  // A full decode is retried in tiles by the caller if it fails.
  VaeDecodeStats decode(const float* latents, std::uint32_t h, std::uint32_t w, std::vector<std::uint8_t>& rgb,
                        std::vector<float>* image = nullptr);
  // ---- Exact band-tiled decode ----
  // The schedule with the smallest workspace for this band height, and among those the fastest. tile is in image
  // pixels, a multiple of 8.
  static VaeTilePlan plan_tiled_min_memory(int base_channels, std::uint32_t h, std::uint32_t w, int tile);
  // A fixed schedule: segments of at most six units, each banded at `tile` (or one band when that covers the
  // segment). Useful for testing every band grid, including a partial final band.
  static VaeTilePlan plan_tiled_uniform(int base_channels, std::uint32_t h, std::uint32_t w, int tile);

  // Byte-identical to decode() in the same mode (fp32/fp16/bf16 activations, matching convolutions, exact integer-sum GroupNorms).
  // Requires a plan for this latent grid and decoder width. set_profile() works as in decode(). The optional
  // `image` receives the fp32 [16h, 16w, 3] output before the clamp, as in decode().
  VaeDecodeStats decode_tiled_exact(const float* latents, std::uint32_t h, std::uint32_t w,
                                    std::vector<std::uint8_t>& rgb, const VaeTilePlan& plan,
                                    std::vector<float>* image = nullptr);

  struct Weights;

 private:
  Device& device_;
  std::unique_ptr<Weights> weights_;
  int base_ = 0;
  hip_vae::Products conv_products_ = hip_vae::Products::kTf32;
  int act_fmt_ = 0;  // activation storage: 0 = fp32, 1 = fp16 bits, 2 = bf16 bits (set with the operand mode)
  bool profile_ = false;
};

}  // namespace mi::hip
