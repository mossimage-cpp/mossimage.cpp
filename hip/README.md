# HIP backend

MOSS-Image's native kernels for AMD GPUs through ROCm, and the same sources through HIP's NVIDIA headers, which run them on a CUDA device with a scalar leaf for checking. It is its own CMake project; the engine's build does not include this directory and nothing here needs CUDA on AMD.

| Target | Compiler | Leaf (`mma_leaf.h`) |
|---|---|---|
| `gfx1100` (RDNA3) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `gfx1200`, `gfx1201` (RDNA4) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `gfx1035` (RDNA2) | ROCm clang | `v_dot2_f32_f16` on bf16 converted to fp16; `v_dot4_i32_i8` for the INT8 GEMM |
| `nvidia` | nvcc with HIP's NVIDIA headers | scalar: the exact bf16*bf16 product in fp32, then add, K ascending |

RDNA2 has no matrix cores and no bf16 instructions: `mma_leaf.h` converts bf16 operands to fp16 on load and accumulates with fp16 dot products, `int8_gemm.hip` uses a VALU tile with `V_DOT4_I32_I8`, and the DiT's attention uses `sage_attn_gfx103x.hip` (int8 Q/K and fp16 V) instead of the WMMA-only int8 SageAttention. The other GEMM kernels keep their usual shapes; runtime dispatch checks the selected device's architecture (`device_is_rdna2()` for the DiT kernels, `Device::info().arch` for the VAE's operand precision).

An AMD build holds a code object for each architecture in `HIP_TARGET` (a CMake list) and runs on those only; a build with `gfx1103` and `gfx1035` produces one exe for both. It needs ROCm 7 at run time (`libamdhip64.so.7`). Every kernel launches workgroups of one wave (32 threads) and depends on that (`mma_leaf.h`). On Windows, clang-cl's `/Fo` makes the ROCm driver duplicate one architecture's code object under every target name, so the CMake compile rule is changed to use `-o`.

## Files

| File | |
|---|---|
| `hip_block.h`, `block_kernels.hip` | One DiT block's kernels and their launches: RMSNorm with modulation, the GEMMs (plain, gated residual, SiTU-GLU), QK norm with RoPE, varlen attention, operand packing. Checked against host replays by the tools below. |
| `gemm_tile.h` | The per-wave GEMM walk the kernel sources share: fragment loads with zero-padded tails, the K-ascending walk over row-major or packed operands, the 2D block order of the tiles, the GEMM kernel and its epilogues (plain, gated residual, bias, bias then residual add), launch helpers. |
| `dit_kernels.h`, `dit_kernels.hip` | The DiT's ops outside its blocks: the linears with a bias, the text refiner's un-modulated RMSNorm, SiLU, SiTU-GLU over a `[hidden \| gate]` GEMM output. |
| `dit.h`, `dit.cpp` | The whole MOSS-Image DiT (below). |
| `mma_leaf.h`, `bf16_leaf.h`, `device_math.h` | The 16x16x16 leaves (gfx11 WMMA, raw gfx11 bf16 WMMA, RDNA2 dot products, NVIDIA scalar) and the bf16 conversions and rounded fp32 operations every kernel uses. |
| `runtime.h` | `mi::hip`: device listing and choice, device and pinned host buffers, streams, events, error checks. |
| `safetensors.h`, `npy.h`, `half.h` | Host-only: the safetensors reader (one file or a sharded directory), the `.npy` reader and writer of the checks, fp16/bf16 conversions. |
| `vae.h`, `vae.cpp` | The FLUX.2 VAE decoder: the final latents [128, h, w] to RGB8 [16h, 16w, 3], at the engine's reference precision except on gfx103x (fp16 operands/activations) and gfx11/gfx12 (bf16 operands/activations). `workspace_bytes(h, w)` is its device memory on the fp32 path (1.50 GiB at 1024x1024, 3.04 GiB at 1456x1456, 6.02 GiB at 2048x2048, plus 113 MB of weights; 16-bit activations roughly halve the ping-pong regions); `decode()` refuses a size the device cannot hold, or past the kernels' limits, before it runs. |
| `vae_kernels.h`, `vae_kernels.hip` | The decoder's kernels. Convolutions accumulate in fp32; gfx103x defaults to one fp16 operand per MAC (`v_dot2_f32_f16`), gfx11/gfx12 to one bf16 operand (native WMMA), elsewhere to fp32/TF32 bf16 pieces. Activation buffers use the same 16-bit format on those paths. Attention follows the operand mode; `MI_HIP_VAE_FP16=0` / `MI_HIP_VAE_BF16=0` restore TF32 and `MI_HIP_VAE_FP16_ACT=0` / `MI_HIP_VAE_BF16_ACT=0` restore fp32 activations. GroupNorm uses the engine's exact statistics. |
| `check_device.sh` | Static check of an AMD device build from its assembly and code object (below). |
| `tools/hip_check.cpp` | The runtime's smoke test. |
| `tools/hip_vae_check.cpp` | The VAE: each kernel against host replays, golden latents against the reference pipeline's PNGs (PSNR, identical bytes), non-square decodes against a host replay of the whole decoder, the size limits. Golden PNG comparisons need zlib (`MI_HIP_VAE_NO_ZLIB` builds without them). |
| `tools/hip_dit_kernel_check.cpp` | `dit_kernels.h`'s ops against host replays. |
| `tools/hip_dit_run.cpp` | The DiT's sampling loop on a golden case, against the official pipeline's outputs. It takes the case's cfg from `cases.json` and the uncond row's skipped blocks from the case's `dit_step0_uncond.skip.json`, and reads them with the engine's JSON reader (`src/util/json.cpp`, compiled in). |
| `text_encoder.h`, `text_encoder.cpp` | The Qwen3.5 text encoder (`last_hidden_state` of the MOSS prompts), in the engine's op order and rounding, streaming its weights from the files one layer at a time. |
| `te_kernels.h`, `te_kernels.hip` | The text encoder's kernels: the Qwen RMSNorms with the text RoPE, the GatedDeltaNet's conv1d + SiLU, gates and chunked gated delta rule, causal GQA attention at head dim 256 with the output gate, the MLP's SiLU product; its GEMMs are the tiled bf16 GEMM by default, or `hip_block`'s on the packed path. |
| `tools/hip_te_kernel_check.cpp` | `te_kernels.h`'s ops against host replays of their op order, and the gated delta rule against an fp64 recurrence. |
| `tools/hip_te_check.cpp` | The whole text encoder against the official pipeline's golden hidden states, gated as the engine's test (`tests/unit/test_qwen35_text.cpp`). |
| `tools/hip_fused_quant_check.cpp` | The fused SiTU-GLU + ConvRot quantizer against the two separate ops (byte-identical). |
| `generate.h`, `generate.cpp` | Text to image: the VAE's weights first, the text encoder, the DiT's sampling loop with CFG and the official Euler step on the host in the engine's order, the VAE decode. The caller supplies the token ids, the initial noise and the schedule; an explicit VRAM budget the run cannot keep to is refused before the first step, with the sizes. A full VAE decode that fails is retried with the exact band decoder; `--vae-tiled` selects it up front, and `MI_HIP_VAE_BAND_TILE_PX` sets its band height. |
| `tools/moss_infer_hip.cpp` | `moss-infer-hip`, the standalone program: `moss-infer generate`'s flags that the backend runs (reference tier), with the engine's tokenizer, initial noise, schedule and PNG writer compiled in (`mossinfer_hip_host`), so a seed gives the same inputs as on CUDA. |

## Building

`ROCM_PATH` is the ROCm 7 root holding `include/hip` and `lib/llvm`, as a CMake cache variable or in the environment. `tools/hip_vae_check` also needs zlib (`zlib1g-dev` or `zlib-devel`).

```sh
cmake -S hip -B build-hip-amd -G Ninja "-DHIP_TARGET=gfx1103;gfx1035"   # any ROCm list; dual-arch builds run on both
cmake --build build-hip-amd -j4
cpack --config build-hip-amd/CPackConfig.cmake  # mossinfer-<version>-linux-hip-x86_64.tar.gz: moss-infer-hip, licences
cmake -S hip -B build-hip-nvidia -G Ninja -DHIP_TARGET=nvidia     # CMAKE_CUDA_ARCHITECTURES, default 89
cmake --build build-hip-nvidia -j4
build-hip-amd/hip_check --list    # the devices HIP sees and whether this build runs on each
build-hip-amd/hip_check           # copies and a known-answer GEMM on the first usable device
build-hip-amd/hip_dit_kernel_check
build-hip-amd/hip_dit_run MODEL/transformer tests/golden g_fridge_1024 [--steps N] [--out FILE.npy] [--dit-budget GiB] [--time-ops] [--expect FILE.npy]
build-hip-amd/moss-infer-hip --model MODEL --prompt "..." -o out.png [--size WxH] [--steps N] [--vram-budget GiB]
```

`hip_dit_run` fails when step 0's cond or uncond prediction is under cosine 0.99988 against the golden one (the CUDA engine's own test's bar) or its rms difference over 0.0155 of the golden rms (what that cosine allows between vectors of equal norm), on a NaN in any step, and, with every step run, when the final latent is under cosine 0.95 or its rms difference over 0.316 of the golden rms. That bound is a gate for gross errors only: 50 steps of sampling amplify small differences (two Vulkan runs differing only in their text encoder, cosine 0.9998 between those, ended 0.9706 apart), and on these golden text-encoder outputs the Vulkan DiT reaches 0.957 (coop) and 0.971 (plain), the CUDA engine 0.992. `--expect` adds an exact comparison of the latent after the steps run with a file `--out` wrote, so a streamed run can be held to a resident run's bits.

Warnings are errors by default (`MI_HIP_WERROR`); the Windows build driver turns it off. An AMD build compiles each kernel source a second time on its own for each architecture, device only and with the library's flags, and runs `check_device.sh` on each assembly and code object: each known kernel must hold its expected WMMA count, use wave size and workgroup size 32, have no scratch, spills or dynamic stack, and the code object must have no undefined symbols. The executables find `libamdhip64` in `$ROCM_PATH/lib` in the build tree; the installed `moss-infer-hip` on `LD_LIBRARY_PATH`, then in `/opt/rocm/lib` (its RUNPATH), then in the system's library paths.

The checks exit 0 when everything passes, 1 on a failure and 2 when no device runs this build (an AMD build on a machine without that GPU, for one). A device whose architecture is not one of the build's targets is listed with the reason and never chosen.

## Runtime

`mi::hip::Device(index)` makes a device current; `index < 0` takes the first device this build runs on. `DeviceBuffer` and `HostBuffer` (page-locked, the source of copies that overlap kernels) free their memory when destroyed; `Stream` is non-blocking; `Event` records, waits across streams and times. Every HIP call that fails throws `std::runtime_error` naming the call and HIP's error string; kernel launches check the launch the same way (`hip_block::` entry points).

The kernel entry points (`hip_block.h`) take device pointers to bf16 bits (`std::uint16_t`) and a stream, launch asynchronously and allocate nothing. Operands are 16-byte aligned with row strides that are multiples of 8 elements.

## DiT

`dit.h` runs MOSSImageTransformer with the op order and rounding of the engine's CUDA forward (`src/models/moss_dit`): the time embedding (a float64 sinusoid on the host, then the bf16 MLP and the shared block and head modulations as linears with M = 1), the video embedder, the text refiner once per prompt (`encode_text`), the 42 blocks over the rows' joint sequences `[video_r | text_r]` concatenated, a block that some rows skip running as one varlen batch per run of the rows that execute it (the uncond row's skip set, {22} or {24} in the golden cases), and the head. The RoPE table is a kernel with the expressions of the engine's (`rope.cu`, device `sinf` and `cosf`). The video tokens and the transpose of the output are host code; CFG and the Euler step belong to the caller (`hip_dit_run` does them in fp32 on the host, in the engine's order). A forward of more joint tokens than the engine's `kMaxRunRows` (61506) is refused, as the engine refuses it.

bf16 blocks run `hip_block`'s packed path (GEMM weights packed into its fragment layout on the host while loading, 9.5 GiB in all, about 216 MiB per block); INT8 ConvRot blocks run `int8_block`'s path (~109 MiB per block). `DitOptions::device_budget` caps the device memory of the weights for both: the blocks past it stay in pinned host memory and are copied into one of two device slots on a second stream, each copy waiting for the block that last used its slot and each block waiting for its copy, so the next block's copy runs while the current one computes. Without a budget every block is resident. A budget below the resident globals and refiner plus the two slots (about 1.03 GiB) is refused. Streaming does not change a bit of the result. The activations at 1024x1024 with a 62-token and a 10-token prompt take about 0.7 GiB (`Dit::activation_bytes`).

The refiner's norm follows torch's reduce order for its row count: a prompt shorter than 16 tokens (the golden uncond prompt has 10) gives more threads per row there, and `hip_block::rms_norm` sums in that order where `rms_norm_modulate` keeps the 32-lane one. The GEMMs accumulate in fp32 with K ascending, as every GEMM here; the engine's cuBLASLt picks its own order, so their sums can differ in the last bit before rounding.

## Checks

Run the standalone checks on a device this build targets. `hip_check` exercises the runtime,
`hip_vae_check` checks the decoder against host replays and optional golden images, and the
`hip_*_check` tools compare individual kernels against fp64/fp32 host references. The
`hip_*_bench` tools report throughput; those numbers depend on the device and toolchain and
are not part of the correctness gate.
