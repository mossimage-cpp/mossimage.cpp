# MOSS-Image HIP backend

Native HIP inference for MOSS-Image on AMD RDNA2/RDNA3/RDNA4 GPUs (`HIP_TARGET`: gfx1035, gfx1100,
gfx1103, gfx1200, gfx1201), built with ROCm 7 (TheRock on Windows). One build can hold several
architectures in a single executable, e.g. `gfx1103;gfx1035`. It is the AMD counterpart of the CUDA
and Vulkan backends.

RDNA2 (gfx1035) has neither matrix cores nor bf16 instructions. There the bf16 GEMMs convert their
operands to fp16 and use `V_DOT2_F32_F16`, the INT8 GEMM uses `V_DOT4_I32_I8` in a VALU tile, and
the DiT attention uses `sage_attn_gfx103x.hip` (int8 Q/K, fp16 V) rather than the WMMA-only int8
SageAttention. The gfx1103 (and other matrix-core) paths are unchanged.

## Features

- `--tier high`: INT8 ConvRot DiT with native gfx11 WMMA intrinsics (RDNA3/4) or RDNA2 dot-product kernels
- Runtime LoRA on all four DiT GEMMs; the delta is fused into the base GEMM's
  fp32 accumulator before bf16 rounding
- INT8 SageAttention-style attention: u8 probabilities on matrix-core devices, int8 Q/K with fp16 V on RDNA2
- Accepts the `moss-infer generate` flag set, UTF-8 prompts, single-file
  `--dit`/`--te`/`--vae` overrides, varlen batches and high-resolution support.
  DiT files with ComfyUI's `model.diffusion_model.` / `diffusion_model.` tensor
  prefixes are accepted as well as the official unprefixed names. StarNodes-style
  INT8 ConvRot files that quantize the globals and text refiner too are accepted:
  those non-block linears are dequantized to bf16 on load using the file's
  `_quantization_metadata`, so no offline rewrite is needed.
  Flags that need CUDA-only machinery are accepted but print a no-effect notice.

## Build (Windows / TheRock)

```bat
hip\build-win.bat
```

The script loads the MSVC environment (via the VS installer), discovers the SDK through
`rocm-sdk path --root` (or `ROCM_PATH`), configures CMake with Ninja, and copies the
executable plus the runtime DLLs into `hip\dist\`. No machine-specific SDK path is
stored. `HIP_TARGET` defaults to `gfx1103;gfx1035`, so one executable runs on both
architectures; override it to build for a different set. Intermediate files stay in
the git-ignored `hip\build-win\`.

Relevant CMake inputs: the TheRock `clang-cl.exe`, `HIP_TARGET`, `ROCM_PATH` pointing at
the SDK's devel root, and `--rocm-device-lib-path=<ROCM_PATH>\lib\llvm\amdgcn\bitcode`.
The runtime DLLs `amdhip64*.dll`, `rocm_kpack.dll` and `amd_comgr.dll` from the SDK must
sit next to the exe; the versioned `amdhip64` name depends on the SDK.

## Run

```bat
moss-infer-hip generate --model <bundle> --dit <file> --te <file> --vae <file> ^
  --prompt "..." --tier high --steps 30 --cfg 4 ^
  [--lora <file> --lora-scale 1.0] -o out.png
```

`--model` supplies the tokenizer and configs; the weight files may be single
`.safetensors` files. Flags requiring CUDA-only machinery are accepted but print
an explicit notice instead of being silently ignored. Launcher scripts are kept
outside the repository.

### VAE tiling

The decoder runs a full-frame decode by default. If that fails, the run retries
in tiles automatically. `--vae-tiled` requests tiling up front. Both use the
exact band decoder: it computes the decoder's GroupNorm statistics over the
whole image, so its output is byte-identical to the full decode and has no
tile-boundary brightness steps. The only tiling control is the band height:

| Variable | Effect |
|---|---|
| `MI_HIP_VAE_BAND_TILE_PX` | band height in image pixels (default 512) |
| `MI_HIP_VAE_FP16` | on gfx1035, round convolution/attention operands to fp16 (default on); `=0` restores the exact TF32 product |
| `MI_HIP_VAE_FP16_ACT` | on gfx1035, keep activation buffers in fp16 too (default on); `=0` restores fp32 activations |
| `MI_HIP_VAE_BF16` | on gfx11/gfx12, round operands to bf16 (default on); `=0` restores the exact TF32 product |
| `MI_HIP_VAE_BF16_ACT` | on gfx11/gfx12, keep activation buffers in bf16 too (default on); `=0` restores fp32 activations |

gfx1035 has no bf16 arithmetic, so the VAE's TF32 operands used to be split into two bf16 pieces and multiplied with
four `v_dot2_f32_f16` products per MAC. By default the decoder keeps one fp16-rounded operand per MAC, which is the
same precision as TF32's 11-bit significand and cuts the VAE's decode time by roughly 50x on an RDNA2 APU. The activation
buffers are fp16 as well on that path, halving the convolutions' global-memory traffic and cutting a 1536x1024 decode
from about 6 s to 4.2 s (30% faster).

gfx11/gfx12 have native bf16 WMMA, so their default is one bf16-rounded operand per MAC instead of four bf16 pieces.
The leaf is the raw `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` (as in `bf16_gemm.hip`), not rocWMMA. Activation
buffers are bf16 as well; in that mode the staging copies the operand bits straight into LDS instead of converting
16-bit -> float -> 16-bit. Measured on gfx1103 at 512x512: the full VAE decode drops from 1.0 s (TF32) to 0.3 s (bf16),
and forcing fp32 activation storage with `MI_HIP_VAE_BF16_ACT=0` gives 0.5 s, so the activation path contributes about
40% of the gain. Accumulation, GroupNorm statistics, the conv_out result and RGB8 rounding stay fp32, so full and
band-tiled decodes remain byte-identical in each mode. The bf16 path changes decoded bytes by at most 8 levels
(mean 0.16) versus TF32; the fp16 path stays within one level. Other devices keep the TF32/fp32 defaults.

The VAE decoder's other switches are `MI_HIP_VAE_ATTN_PIECES=3` (exact 3-piece attention; effective only in the
NVIDIA check build — every supported AMD device has a one-product operand path) and `MI_HIP_VAE_PROFILE` (per-op timings).

## Implementation notes

- `int8_leaf.h` uses raw `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`
  intrinsics, not rocWMMA: each lane holds a whole 16-byte K-step, and
  accumulator element `e` is row `2*e + lane/16`, column `lane % 16`.
- On matrix-core devices the INT8 and bf16 GEMMs share an LDS-staged,
  software-pipelined 128x128 tile; shallow K uses a 512-thread 2x2 fragment grid,
  deep K a 128-thread 4x4 grid. On RDNA2 the bf16 GEMM uses a 64x64 / 256-thread
  tile and the INT8 GEMM a 128x128 / 256-thread VALU tile.
- Attention evaluates `Sᵀ = K·Qᵀ` and `Oᵀ = Vᵀ·Pᵀ` in 128-query / 64-key tiles,
  keeping u8 probabilities in registers so the online softmax is scalar per
  lane. Q/K/V quantizers derive all varlen offsets from `cu_seqlens` on device.
  The V padding is left uninitialized: out-of-range keys get zero probability,
  so those bytes never contribute to PV.
- The text encoder's int8 ConvRot weights are dequantized to host bf16 once at
  construction. The reference dense Hadamard multiply is O(g^2) and dominates
  startup; `dequant_fast.h` applies the same H4-based transform as radix-4
  butterfly stages (O(g log g)), writes bf16 directly and processes tensors on
  several threads. The fast transform differs from the reference at a small
  fraction of bf16 weight values; set `MI_HIP_TE_EXACT_DEQUANT` (any value) for
  bit-exact but much slower weights. Projection GEMMs use the LDS-staged tiled
  kernel on row-major weights by default; `MI_HIP_TE_PACKED_GEMM` (any value
  except `0`) selects the original packed rocWMMA path, `0` keeps the tiled
  kernel. "text encoder load" and "text
  encoder" are reported separately.
  The int8 ConvRot layer weights default to staying int8 on the device:
  `te_dequant.hip` runs the same radix-4 transform there as the host path and
  writes bf16 directly into the per-layer buffer, avoiding the second host bf16
  image and its copy. This is bit-identical to the host dequantization;
  `MI_HIP_TE_DEVICE_DEQUANT=0` restores the host path and uses one fewer
  layer-sized device buffer.
- The VAE decoder's mid attention packs Q/K/V/P into 2 TF32 bf16 pieces on the
  TF32 path (3 GEMM products instead of 6), with a small decoded-image difference
  from the fp32-accurate path. The AMD defaults are one fp16 operand per product
  with fp16 buffers on gfx103x and one bf16 operand with bf16 buffers on
  gfx11/gfx12; the switches above select the other modes.
- The DiT attention output head stride is the head dimension (128), not the
  full residual width; using the latter corrupts adjacent token rows.
- The LoRA delta must be added before the base GEMM's bf16 rounding, not to the
  rounded output, to match the CUDA side branch's magnitude.
- Grids that scale with token count are chunked so large images stay within
  driver limits; partial M tiles are padded and launch errors are checked.

## Checks and benchmarks

`CMakeLists.txt` builds the standalone checks and benchmarks by default
(`MOSSINFER_HIP_TOOLS=ON`); the Windows `build-win.bat` currently configures with
them off. On a build with them enabled:

- `hip_int8_gemm_check`, `hip_bf16_gemm_check`, `hip_int8_attn_probe`,
  `hip_attn_probe` — compare kernels against host references
- `hip_int8_gemm_bench`, `hip_bf16_gemm_bench`, `hip_rotate_quantize_bench`,
  `hip_lora_info`, `hip_te_profile`, `hip_dequant_fast_check` — report kernel
  throughput, LoRA branch shapes and text-encoder timing/accuracy
- `hip_fused_quant_check` — checks the fused SiTU-GLU + ConvRot quantizer against the two separate ops
- The probe tools' debug-only switches: `MI_ATTN_QKV` (`hip_attn_probe`) loads a raw qkv dump;
  `MI_I8ATTN_BACKEND`, `MI_I8ATTN_BENCH`, `MI_I8ATTN_DEBUG`, `MI_I8ATTN_DEQ` (`hip_int8_attn_probe`) select a backend,
  a throughput loop and intermediate dumps. They affect only those tools.
- `MI_HIP_OP_TIME` (any value) prints a per-op DiT timing breakdown
