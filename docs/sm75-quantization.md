# Quantized models on sm_75: status and plan

The mainline engine runs three quality tiers. On sm_75 only the bf16 `reference` tier is *available without extra
files* today; the quantized tiers work when the models are already quantized, and the pieces needed to make them
first-class (and to produce the files) are listed below with a plan.

## What the tiers mean

| Tier | Weights | Attention | What the loaded DiT must be |
|---|---|---|---|
| `reference` | bf16, exact | exact | a bf16 DiT and text encoder |
| `high` | INT8 ConvRot (W8A8), optional quantized text encoder | 8-bit (`q8_fp16`/`q8_fp8`) | an INT8 ConvRot DiT **file**, or a bf16 DiT *and* 8-bit attention |
| `turbo` | INT4 ConvRot (`convrot_w4a4`) or NVFP4 | 8-bit | an INT4/NVFP4 DiT file (NVFP4 needs an RTX 50) |

`planner::resolve_tier` (src/planner/planner.cpp) enforces that: `high` with exact attention on a bf16 DiT is "the
Reference configuration" and is refused, and on sm_75 the missing 8-bit attention kernels make `exact_when_no_q8` pick
exact attention, so `high` needs an INT8 DiT file. `turbo` only accepts a 4-bit file.

## What already works on sm_75

Measured on the RTX 2080 8 GiB with the SFT bundle (`balanced`), this branch, after the attention and VAE work:

| Configuration | DiT file | Denoise, 256^2, 8 steps | Denoise, 1024^2, 2 steps | Device peak |
|---|---|---|---|---|
| `reference` | bf16 bundle | 7.50 s (0.94 s/step) | 8.60 s (4.3 s/step) | 6.37 GiB |
| `high` | INT8 ConvRot, 5.04 GB | **0.99 s (0.12 s/step)** | **5.18 s (2.54 s/step)** | 5.27 GiB |
| `turbo` | INT4 ConvRot, 2.83 GB | 1.14 s (0.14 s/step) | 5.16 s (2.57 s/step) | 3.06 GiB |

- The INT8 GEMMs are CUTLASS Sm75 m8n8k16 kernels with the sm_80 epilogues (dequantize+bias, gated residual,
  SiTU-GLU) under `src/kernels/sm75/int8_gemm_sm75.cu`; one tile configuration (64x128x64) fits Turing's 64 KiB of
  shared memory, so forcing another tile is refused (`check_sm75_tile`).
- INT4 weights stay packed and are unpacked into the INT8 GEMM (`quant4/w4a8_decode.cu`), so `turbo` is correct on
  Turing (about 1.5x the INT8 step in the planner's price, not a native 4-bit MMA).
- Exact attention runs with every tier; the quantized tiers do not need 8-bit attention on sm_75.
- `moss-infer quantize` runs on an 8 GiB Turing card: the 5B DiT became a 5.04 GB INT8 ConvRot file in 49 s
  (252 GEMMs quantized, 0 kept bf16).
- `moss-infer quantize --format int4` (added for P1) writes the Turbo tier's `convrot_w4a4` format: 2.83 GB in 55 s
  with `--recipe rtn`, 283 s with the default `--recipe mse` (its clipping-ratio search). Against a published community
  `convrot_w4a4` file the quantized weights match at cosine 0.9993 (dequantized; the codes differ only by the 256-column
  rotation GEMM's rounding), the bf16 tensors are bit-identical and the file loads and runs `tier turbo` on the RTX
  2080 (0.16 s/step at 256^2, 3.9 GiB peak).
- The VAE stays fp32 SIMT either way (see `docs/sm75.md`).

## Gaps

**G1 - INT4 writer (done, P1).** `moss-infer quantize --format int4` now writes the Turbo tier's `convrot_w4a4`
format: packed W4 (`[N, K/2]`, even column the low nibble), a per-row f32 scale, 256-column rotation groups, and the
file-level `_quantization_metadata` (`{"format_version": "1.0", "layers": {<module>: {"format": "convrot_w4a4",
"convrot_groupsize": 256, "quant_group_size": 64}}}`). The scale rule is comfy-kitchen's clipping-ratio search
(`--recipe mse`, the default) or absmax (`--recipe rtn`); LoRA merging and `--keep-bf16` work as for INT8. K must be a
multiple of 256.

**G2 - `high` with the bf16 bundle is refused on sm_75.** The tier is defined as INT8 GEMMs *plus* 8-bit attention;
sm_75 has no 8-bit attention, so the resolver falls back to exact attention and then refuses the combination. A
Turing user has to quantize the DiT first (or download an INT8 file) to use `high`.

**G3 - 8-bit attention (done, P2).** `src/kernels/sm75/attention_q8_sm75.cu` implements `q8_fp16` on Turing:
per-row INT8 q/k (a `q8_quantize_rows` pre-pass, v converted to fp16), QK^T with the m8n8k16 INT8 MMA, the exact
kernel's fp32 online softmax and fp16 P·V. It is wired through `attention_q8.cu`'s three entry points, so
`tier high` now runs on a **bf16 DiT** (`tier high: DiT bf16, q8_fp16 attention`) — verified on the RTX 2080 at
256^2 (output cosine 0.9986 against the exact path) and 1024^2. Speed: at 256^2 and 1024^2 it is within noise of the
exact path (0.96 vs 0.94 s/step; 4.39 vs 4.30 s/step), because Turing's INT8 MMA (`m8n8k16`) does the same MACs per
instruction as the fp16 `m16n8k8` it replaces — Ampere's `m16n8k32` is what doubles INT8 throughput, and Turing does not
have it. The 8-bit mode is therefore a memory-footprint/bandwidth path on Turing, not a compute accelerator;
`q8_fp8` stays refused (no e4m3 MMA).

**G4 - LoRA on quantized DiTs (done, P4).** On sm_75 every adapter is merged (the side branches are sm_80+ kernels).
Verified on the RTX 2080: `reference + --lora` merges into bf16; `high + --lora` merges into the INT8 ConvRot weights
through `merge_requantize_int8` and reports "merged into INT8 (requantization error)" per file; `turbo + --lora` is
refused up front with a message naming the reason (INT4 ConvRot has no merge path; Turing has no side branches) and
suggesting `tier high` or `tier reference`.

**G5 - the app has to offer it.** MossStudio decides what to download and which tiers to show; a Turing card should be
able to run `quantize` (engine CLI) or fetch INT8/INT4 files, and the tier selector should enable `high`/`turbo` when
one is loaded.

## Plan

**P1 - INT4 output in `quantize` (done).** `--format int4 --recipe mse|rtn` writes a `convrot_w4a4` DiT the same card
can run with `--tier turbo`; the produced file is 2.83 GB and matches a published file's weights at cosine 0.9993.

**P2 - Turing 8-bit attention (`q8_fp16`) (done).** `attention_q8_sm75.cu` (per-row INT8 q/k, m8n8k16 QK^T, the
exact kernel's softmax and fp16 P·V) is wired into `attention_q8.cu`; `tier high` runs on a bf16 DiT and matches the
exact path at cosine 0.9986. The expected 1.5-2x attention speedup did not materialize: Turing's INT8 MMA is
`m8n8k16`, the same MACs per instruction as the fp16 `m16n8k8` it replaces (Ampere's `m16n8k32` is the 2x part), so
8-bit attention is a bandwidth/footprint path on this architecture, not a compute one.

**P2 design notes (from the code, for the implementation).** The whole 8-bit path funnels through three functions in
`src/kernels/attention_q8/attention_q8.cu` (`supported`, `attention_q8_workspace_bytes`, `attention_q8`), so a Turing
implementation only has to answer those three for `kmInt8Fp16`/`kInt8Fp16F32Acc`; `moss_dit` and `resolve_tier` need no
change beyond `supported()` turning true (tier `high` then picks `q8_fp16` on a bf16 DiT, because `kTierAttention` is
`{q8_fp8, q8_fp16}` and picks the first supported one). Workspace on sm_75 should use its own layout:
`q_i8 [total_q][heads][128]`, `k_i8 [total_k][kv][128]`, `v_f16`, `q_scale [total_q][heads]`, `k_scale [total_k][kv]`.
The kernel can reuse the exact kernel's skeleton (`attention_sm75.cu`: 64-query CTA, 4 warps, 16 rows per warp, K/V
tiles in shared, online softmax): only the score mainloop changes. Turing's INT8 MMA is `m8n8k16`
(`mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32`, see `third_party/cutlass/include/cutlass/arch/mma_sm75.h`):
A = 4 consecutive int8 per lane (row = lane/4, dims = 16*(chunk) + 4*(lane%4)), B = 4 consecutive int8 per lane
(key = lane/4, same dims), C = 2 int32 per lane (row = lane/4, keys = 8*subtile + 2*(lane%4), +1). With two row groups
per warp (rows r and r+8) this maps exactly onto the existing `s[t][0..3]` register block, so only the fragment loads,
the mma and the dequantization (`s = acc * q_scale[row] * k_scale[key] * scale`) are new. `q` is quantized per row and
`k` per row (SageAttention smooths K by its per-sequence mean; optional here); `v` becomes fp16 and the P·V path stays
`m16n8k8`. The fused QK-norm/RoPE pre-pass of `attention_q8` is not worth porting: `dit_block_quant.cpp` already has a
separate-kernel path (`fused_norm_rope` false, same bits), so the sm_75 q8 branch needs that condition forced off for
Turing.
**P3 - or, if P2 waits: a sm_75 tier rule.** Let `high` on cc 7.5 quantize the block GEMMs at load
(`LoadOptions::int8_block_gemms` currently only follows an INT8 file) and keep exact attention, documented as a
sm_75-specific definition of `high`. Small (planner + load options), but it changes what `high` means on Turing, so
P2 is the cleaner long-term answer.

**P4 - LoRA on quantized tiers (done).** `high + --lora` works on sm_75 (host requantize merge, warning as on sm_80);
`turbo + --lora` refuses with the reason and the working alternatives.

**P5 - docs and app.** Update `docs/sm75.md`'s limits with the outcome, add the INT4 production step and the measured
tier table here, and tell MossStudio which files/tiers a cc 7.5 device can use.

## Test matrix

- `sm75_check int8` (existing) plus a `quantize` -> load -> generate round trip for INT8 and INT4 at 256^2 and 1024^2.
- `high`/`turbo` with `--attention exact` and (after P2) `q8_fp16`: same prompt and seed, check the report's tier line
  and `quality.gpu_arch`.
- LoRA: `--tier reference` (merged, works today), `--tier high --lora` (P4), `--tier turbo --lora` (clear refusal).
- Regression: `sm75_check` all cases, and the OFF build (`MOSSINFER_SM75=OFF`) still compiles and passes.
