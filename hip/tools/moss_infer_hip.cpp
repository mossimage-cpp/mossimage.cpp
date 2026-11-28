// moss-infer-hip: text to image on the HIP backend, a standalone program for AMD GPUs through ROCm (Linux). It needs
// neither CUDA nor an NVIDIA driver, only ROCm 7's runtime; the binary holds the kernels of each architecture it was built
// for. Its flags are those of `moss-infer generate` that the backend runs, checked as the engine checks them; the
// tokenizer, the seed's initial noise, the schedule and the PNG writer are the engine's own sources, so a seed gives the
// same inputs as on CUDA. Exit status: 0 done, 1 failed, 2 a usage error or an unsupported request.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The engine's headers before the HIP ones (generate.h): those define __noinline__ as a macro, which libstdc++ uses.
#include "mossinfer/mossinfer.h"
#include "pipeline/png_writer.h"
#include "pipeline/samplers/initial_noise.h"
#include "pipeline/samplers/schedules.h"
#include "tokenizer/prompt_tokens.h"
#include "tokenizer/qwen_bpe.h"
#include "util/json.h"
#include "util/path.h"
#include "util/regular_file.h"

#include "generate.h"
#include "hip_block.h"
#include "npy.h"
#include "runtime.h"

namespace {

namespace fs = std::filesystem;
using namespace mi;

// pipeline.h's request bounds (kMaxSide, kMaxPixels, kMaxSteps, kMaxSeed, kMaxVramBytes), which validate() applies;
// pipeline.h itself needs the CUDA headers.
constexpr int kMaxSide = 4096;
constexpr std::int64_t kMaxPixels = 4'200'000;
constexpr int kMaxSteps = 10000;
constexpr std::int64_t kMaxSeed = std::int64_t{1} << 53;
constexpr double kMaxVramBytes = static_cast<double>(std::uint64_t{1024} << 30);

constexpr const char* kUsage =
    "usage: moss-infer-hip [generate] --model <bundle_dir> --prompt <text> [-o <out.png>] [--latent-out <latents.npy>]\n"
    "    [--params <parameters.json>] [--json-report <report.json>] [--negative <text>] [--size <W>x<H>]\n"
    "    [--width N] [--height N] [--steps N] [--cfg X] [--shift X] [--seed N|-1] [--skip-layers \"22\"]\n"
    "    [--slg-layers \"\"] [--slg-scale X] [--slg-start X] [--slg-stop X] [--max-seq N]\n"
    "    [--sampler euler|ab2|euler_ancestral|heun|dpmpp_2m|dpmpp_2m_sde|er_sde|res_multistep]\n"
    "    [--scheduler moss|simple|normal|karras|exponential|beta|sgm_uniform] [--flow-multiplier X]\n"
    "    [--noise-layout official|batch_stream] [--noise-source philox|philox_device|brownian_tree]\n"
    "    [--eta X] [--s-noise X] [--dit <file>] [--te <file>] [--vae <file>]\n"
    "    [--lora <file> [--lora-scale S]]... [--lora-strict] [--lora-placement default|merge|side_branch]\n"
    "    [--tier reference|high] [--mode min_memory|balanced|max_speed] [--attention auto|exact|q8*]\n"
    "    [--dit-blocks fused|fused_graphs|op_by_op] [--vram-budget GiB] [--vram-reserve MiB] [--vae-tiled]\n"
    "    [--host-ram-budget GiB] [--cache-dir <dir>] [--te-host-pin auto|on|off]\n"
    "    [--direct-io | --buffered-io] [--then-prompt <text>]...\n"
    "    [--cfg-truncation X] [--step-cache off|fbcache] [--step-cache-threshold X]\n"
    "    [--preview <png> [--preview-every N]] [--init-image <img> [--strength X] [--mask <img>] [--resize crop|fit|stretch]]\n"
    "    [--hires <W>x<H>|<scale> [--hires-strength X] [--hires-steps N]] [--distill <dir>] [--time-embedding <file>]\n"
    "    [--control <file> [--control-scale S] [--control-image kind=<img>] ...] [--reference <img>]... [--reference-max-edge N]\n"
    "    [--device N] [--no-metadata] [--quiet] [-h|--help] [--version] [--list-devices]\n"
    "Text to image with MOSS-Image on an AMD GPU through native HIP. All `moss-infer generate` flags are accepted;\n"
    "the defaults match it (1024x1024, 50 steps, cfg 3, shift 5, seed 42, skip layers 22). At least one of -o /\n"
    "--latent-out is required; without -o the VAE decode is skipped. Options that need CUDA-only machinery (planner\n"
    "modes, GGUF cache, direct I/O, image conditioning, LoRA merging) print a clear notice instead of being silently\n"
    "ignored. --device picks a HIP device (default: the first usable one). Progress goes to stderr one line per event.\n";

struct Args {
  fs::path model, out_png, out_latent, dit, te, vae, lora, json_report, params;
  std::string prompt, negative;
  float lora_scale = 1.0f;
  std::string lora_placement;
  bool lora_strict = false;
  std::string tier = "high";
  // The rest of the `moss-infer generate` option surface. Some are planner/kernel choices with no counterpart in this
  // backend (they are recorded and reported); the ones that change the sampling maths are passed to generate().
  std::string sampler = "euler", scheduler = "moss", noise_layout = "official", noise_source = "philox";
  std::string mode = "max_speed", attention = "auto", dit_blocks = "fused", cache_dir, te_host_pin = "auto";
  std::string init_image, mask, resize = "crop", hires, distill, time_embedding, dataspace;
  double flow_multiplier = 1.0, eta = 0.0, s_noise = 1.0;
  double slg_scale = 0.0, slg_start = 0.0, slg_stop = 1.0, cfg_truncation = 1.0;
  double step_cache_threshold = 0.0;
  std::vector<int> slg;
  std::vector<std::string> then_prompts, control_files, control_images, references;
  double control_scale = 1.0, control_strength = 1.0, control_start = 0.0, control_end = 1.0;
  int reference_max_edge = 512, preview_every = 1;
  fs::path preview;
  float strength = 0.75f, hires_strength = 0.45f;
  int hires_steps = 0, vram_reserve = 0;
  double host_ram_budget = 0;
  std::string step_cache = "off";
  bool direct_io = false, buffered_io = false, vae_tiled = false;
  bool have_prompt = false, metadata = true, quiet = false, list = false, help = false, version = false;
  int width = 1024, height = 1024, steps = 50, max_seq = 4096, device = -1;
  double cfg = 3.0, shift = 5.0;
  std::uint64_t vram_budget = 0;
  std::int64_t seed = 42;
  std::vector<int> skip = {22};
};

[[noreturn]] void usage(const std::string& what) { throw std::invalid_argument(what); }

double number(const std::string& flag, const std::string& v) {
  std::size_t used = 0;
  double d = 0;
  try {
    d = std::stod(v, &used);
  } catch (const std::exception&) {
    usage(flag + " takes a number, got " + v);
  }
  if (used != v.size() || !std::isfinite(d)) usage(flag + " takes a finite number, got " + v);
  return d;
}

std::int64_t integer(const std::string& flag, const std::string& v) {
  std::size_t used = 0;
  long long i = 0;
  try {
    i = std::stoll(v, &used);
  } catch (const std::exception&) {
    usage(flag + " takes an integer, got " + v);
  }
  if (used != v.size()) usage(flag + " takes an integer, got " + v);
  return i;
}

int int32(const std::string& flag, const std::string& v) {
  const std::int64_t i = integer(flag, v);
  if (i < std::numeric_limits<int>::min() || i > std::numeric_limits<int>::max()) usage(flag + " is out of range");
  return static_cast<int>(i);
}

std::vector<int> layers(const std::string& v) {
  std::vector<int> out;
  std::size_t at = 0;
  while (at < v.size()) {
    const std::size_t comma = v.find(',', at);
    std::string item = v.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
    item.erase(0, item.find_first_not_of(' '));
    item.erase(item.find_last_not_of(' ') + 1);
    if (!item.empty()) {
      const int l = int32("--skip-layers", item);
      if (l < 0 || l >= hip::kDiTLayers) usage("--skip-layers takes block indices in [0, 41]");
      out.push_back(l);
    }
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  return out;
}

std::string read_file(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + util::path_utf8(p));
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// The subset of `moss-infer generate --params` this backend acts on. The file is applied once, before every
// command-line flag, so flags override these values regardless of argument order, exactly as in the CUDA CLI.
void apply_params(Args& a, const json::Value& doc) {
  if (!doc.is_object()) usage("--params: expected a JSON object");
  const json::Value* params = doc.find("parameters");
  if (params != nullptr) {
    if (!params->is_object()) usage("--params: \"parameters\" must be an object");
  } else {
    params = &doc;
  }
  const json::Value& p = *params;
  const auto str = [&](const char* key, std::string& dst) {
    const json::Value* v = p.find(key);
    if (v == nullptr || v->is_null()) return;
    if (!v->is_string()) usage(std::string(key) + " must be a string");
    dst = v->as_string();
  };
  const auto num = [&](const char* key, double& dst) {
    const json::Value* v = p.find(key);
    if (v == nullptr || v->is_null()) return;
    if (!v->is_number()) usage(std::string(key) + " must be a number");
    dst = v->as_double();
  };
  const auto integer = [&](const char* key, std::int64_t& dst) {
    const json::Value* v = p.find(key);
    if (v == nullptr || v->is_null()) return;
    if (!v->is_int()) usage(std::string(key) + " must be an integer");
    dst = v->as_int();
  };
  str("prompt", a.prompt);
  if (!a.prompt.empty()) a.have_prompt = true;
  str("negative_prompt", a.negative);
  // The engine checks the int range; a bare int64 -> int cast would turn an out-of-range value into a valid small one.
  const auto checked_int = [&](const char* key, int& dst) {
    std::int64_t i = dst;
    integer(key, i);
    if (i < std::numeric_limits<int>::min() || i > std::numeric_limits<int>::max())
      usage(std::string(key) + " is out of range.");
    dst = static_cast<int>(i);
  };
  checked_int("height", a.height);
  checked_int("width", a.width);
  checked_int("steps", a.steps);
  if (const json::Value* s = p.find("seed"); s != nullptr) {
    if (s->is_null())
      a.seed = -1;  // as in pipeline::request_from_parameters: a null seed is a random seed
    else {
      if (!s->is_int()) usage("seed must be an integer");
      a.seed = s->as_int();
    }
  }
  checked_int("max_sequence_length", a.max_seq);
  num("cfg", a.cfg);
  num("shift", a.shift);
  num("slg_scale", a.slg_scale);
  num("slg_start", a.slg_start);
  num("slg_stop", a.slg_stop);
  num("cfg_truncation", a.cfg_truncation);
  num("step_cache_threshold", a.step_cache_threshold);
  str("step_cache", a.step_cache);
  const auto layer_list = [&](const char* key) {
    const json::Value* v = p.find(key);
    if (v == nullptr || v->is_null()) return std::string();
    if (!v->is_array()) usage(std::string(key) + " must be an array");
    std::string s;
    for (const json::Value& x : v->as_array()) {
      if (!x.is_int()) usage(std::string(key) + " must contain integers");
      if (!s.empty()) s += ',';
      s += std::to_string(x.as_int());
    }
    return s;
  };
  if (p.find("uncond_skip_layers") != nullptr) a.skip = layers(layer_list("uncond_skip_layers"));
  else if (p.find("slg_layers") != nullptr) a.skip = layers(layer_list("slg_layers"));
}

// Human/JSON-readable name of the VAE path that ran, with the reason when tiling was a fallback.
std::string vae_decode_label(const hip::GenerateResult& r) {
  using P = hip::GenerateResult::VaeDecodePath;
  switch (r.vae_decode_path) {
    case P::kFull: return "full";
    case P::kTiledRequested: return "tiled (--vae-tiled)";
    case P::kTiledBudget: return "tiled (VRAM budget)";
    case P::kTiledFallback: return "tiled (full decode failed: " + r.vae_decode_note + ")";
    case P::kNone: return "none";
  }
  return "unknown";
}

std::string json_escape(const std::string& s) {
  std::string out = "\"";
  for (const unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          out += b;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out + '"';
}

// The sampler names the engine accepts (pipeline::parse_sampler lives in a CUDA source not linked here).
bool valid_sampler(const std::string& v) {
  return v == "euler" || v == "ab2" || v == "euler_ancestral" || v == "heun" || v == "dpmpp_2m" ||
         v == "dpmpp_2m_sde" || v == "er_sde" || v == "res_multistep" || v == "dataspace";
}

// pipeline::vram_option_bytes: GiB to bytes, with its bounds and messages.
std::uint64_t vram_bytes(const std::string& flag, double gib) {
  if (!std::isfinite(gib) || gib <= 0) usage(flag + " must be a finite positive number.");
  const double bytes = gib * static_cast<double>(std::uint64_t{1} << 30);
  if (!(bytes <= kMaxVramBytes)) usage(flag + " must be at most 1024 GiB.");
  if (bytes < static_cast<double>(std::uint64_t{1} << 20)) usage(flag + " must be at least 1 MiB.");
  return static_cast<std::uint64_t>(bytes);
}

// Apply one command-line flag. --params is handled separately, after all flags have been collected.
void apply_flag(Args& a, const std::string& f, const std::string& v) {
  if (f == "--no-metadata") a.metadata = false;
  else if (f == "--direct-io") a.direct_io = true;
  else if (f == "--buffered-io") a.buffered_io = true;
  else if (f == "--lora-strict") a.lora_strict = true;
  else if (f == "--quiet") a.quiet = true;
  else if (f == "--list-devices") a.list = true;
  else if (f == "--vae-tiled") a.vae_tiled = true;
  else if (f == "--json-report") a.json_report = util::path_from_utf8(v);
    else if (f == "--model") a.model = util::path_from_utf8(v);
    else if (f == "--dit") a.dit = util::path_from_utf8(v);
    else if (f == "--te") a.te = util::path_from_utf8(v);
    else if (f == "--vae") a.vae = util::path_from_utf8(v);
    else if (f == "--lora") {
      if (!a.lora.empty()) usage("--lora may be given only once by this backend");
      a.lora = util::path_from_utf8(v);
    }
    else if (f == "--lora-scale") a.lora_scale = static_cast<float>(number(f, v));
    else if (f == "--lora-placement") a.lora_placement = v;
    else if (f == "--prompt") a.prompt = v, a.have_prompt = true;
    else if (f == "--negative") a.negative = v;
    else if (f == "-o" || f == "--output") a.out_png = util::path_from_utf8(v);
    else if (f == "--latent-out") a.out_latent = util::path_from_utf8(v);
    else if (f == "--size") {
      const std::size_t x = v.find('x');
      if (x == std::string::npos) usage("--size takes <W>x<H>");
      a.width = int32("--size", v.substr(0, x));
      a.height = int32("--size", v.substr(x + 1));
    }
    else if (f == "--width") a.width = int32(f, v);
    else if (f == "--height") a.height = int32(f, v);
    else if (f == "--steps") a.steps = int32(f, v);
    else if (f == "--cfg") a.cfg = number(f, v);
    else if (f == "--shift") a.shift = number(f, v);
    else if (f == "--seed") a.seed = integer(f, v);
    else if (f == "--skip-layers") a.skip = layers(v);
    else if (f == "--slg-layers") a.slg = layers(v);
    else if (f == "--slg-scale") a.slg_scale = number(f, v);
    else if (f == "--slg-start") a.slg_start = number(f, v);
    else if (f == "--slg-stop") a.slg_stop = number(f, v);
    else if (f == "--max-seq") a.max_seq = int32(f, v);
    else if (f == "--device") a.device = int32(f, v);
    else if (f == "--vram-budget") a.vram_budget = vram_bytes(f, number(f, v));
    else if (f == "--vram-reserve") a.vram_reserve = int32(f, v);
    else if (f == "--host-ram-budget") a.host_ram_budget = number(f, v);
    else if (f == "--sampler") {
      if (!valid_sampler(v)) usage("--sampler takes euler, ab2, euler_ancestral, heun, dpmpp_2m, dpmpp_2m_sde, er_sde, res_multistep or dataspace");
      if (v != "euler") usage("--sampler: this backend implements only euler");
      a.sampler = v;
    }
    else if (f == "--scheduler") {
      try { pipeline::parse_scheduler(v); } catch (const std::exception&) { usage("--scheduler takes moss, simple, normal, karras, exponential, beta or sgm_uniform"); }
      a.scheduler = v;
    }
    else if (f == "--flow-multiplier") a.flow_multiplier = number(f, v);
    else if (f == "--noise-layout") {
      try { pipeline::parse_noise_layout(v); } catch (const std::exception&) { usage("--noise-layout takes official or batch_stream"); }
      a.noise_layout = v;
    }
    else if (f == "--noise-source") {
      if (v != "philox" && v != "philox_device" && v != "brownian_tree") usage("--noise-source takes philox, philox_device or brownian_tree");
      a.noise_source = v;
    }
    else if (f == "--eta") a.eta = number(f, v);
    else if (f == "--s-noise") a.s_noise = number(f, v);
    else if (f == "--cfg-truncation") a.cfg_truncation = number(f, v);
    else if (f == "--step-cache") a.step_cache = v;
    else if (f == "--step-cache-threshold") a.step_cache_threshold = number(f, v);
    else if (f == "--mode") a.mode = v;
    else if (f == "--attention") a.attention = v;
    else if (f == "--dit-blocks") a.dit_blocks = v;
    else if (f == "--cache-dir") a.cache_dir = v;
    else if (f == "--te-host-pin") a.te_host_pin = v;
    else if (f == "--preview") a.preview = util::path_from_utf8(v);
    else if (f == "--preview-every") a.preview_every = int32(f, v);
    else if (f == "--init-image") a.init_image = v;
    else if (f == "--strength") a.strength = static_cast<float>(number(f, v));
    else if (f == "--mask") a.mask = v;
    else if (f == "--resize") a.resize = v;
    else if (f == "--hires") a.hires = v;
    else if (f == "--hires-strength") a.hires_strength = static_cast<float>(number(f, v));
    else if (f == "--hires-steps") a.hires_steps = int32(f, v);
    else if (f == "--distill") a.distill = v;
    else if (f == "--time-embedding") a.time_embedding = v;
    else if (f == "--dataspace") a.dataspace = v;
    else if (f == "--then-prompt") a.then_prompts.push_back(v);
    else if (f == "--control") a.control_files.push_back(v);
    else if (f == "--control-scale") a.control_scale = number(f, v);
    else if (f == "--control-image") a.control_images.push_back(v);
    else if (f == "--control-strength") a.control_strength = number(f, v);
    else if (f == "--control-start") a.control_start = number(f, v);
    else if (f == "--control-end") a.control_end = number(f, v);
    else if (f == "--reference") a.references.push_back(v);
    else if (f == "--reference-max-edge") a.reference_max_edge = int32(f, v);
    else if (f == "--tier") {
      // reference: bf16 weights; high: the INT8 ConvRot DiT (int8 x int8 -> int32 -> bf16), auto-detected per block from the
      // model's safetensors dtypes. The other tiers are CUDA-only.
      if (v != "reference" && v != "high") usage("moss-infer-hip runs --tier reference or high");
      a.tier = v;
    }
    else usage("unknown or unsupported flag " + f);
}

Args parse(const std::vector<std::string>& argv) {
  Args a;
  // Accept the CUDA CLI's `generate` / `check` subcommand prefix so the same launcher works with either binary.
  std::size_t first = 1;
  if (first < argv.size() && (argv[first] == "generate" || argv[first] == "check")) ++first;
  std::vector<std::pair<std::string, std::string>> flags;
  for (std::size_t i = first; i < argv.size(); ++i) {
    const std::string& f = argv[i];
    if (f == "-h" || f == "--help") {
      a.help = true;
      return a;
    }
    if (f == "--version") {
      a.version = true;
      return a;
    }
    if (f == "--no-metadata" || f == "--direct-io" || f == "--buffered-io" || f == "--lora-strict" ||
        f == "--quiet" || f == "--list-devices" || f == "--vae-tiled") {
      flags.emplace_back(f, "");
      continue;
    }
    if (i + 1 >= argv.size()) usage(f + " needs a value");
    flags.emplace_back(f, argv[++i]);
  }
  // Like the CUDA CLI: read --params once, then apply every flag in command-line order, so a flag overrides the file
  // regardless of where --params appears in argv.
  for (const auto& [f, v] : flags)
    if (f == "--params") a.params = util::path_from_utf8(v);
  if (!a.params.empty()) apply_params(a, json::parse(read_file(a.params)));
  for (const auto& [f, v] : flags)
    if (f != "--params") apply_flag(a, f, v);
  // Planner and kernel-choice flags the HIP backend does not have. They are accepted (CUDA launchers can drive this
  // binary unchanged) but reported, rather than silently changing nothing.
  const auto note = [](const char* flag) {
    std::fprintf(stderr, "moss-infer-hip: note: %s has no effect on this backend\n", flag);
  };
  if (a.mode != "max_speed") note("--mode");
  if (a.attention != "auto") note("--attention");
  if (a.dit_blocks != "fused") note("--dit-blocks");
  if (!a.cache_dir.empty()) note("--cache-dir");
  if (a.step_cache != "off") note("--step-cache");
  if (a.step_cache_threshold != 0) note("--step-cache-threshold");
  if (a.direct_io || a.buffered_io) note("--direct-io/--buffered-io");
  if (a.vram_reserve != 0) note("--vram-reserve");
  if (a.host_ram_budget != 0) note("--host-ram-budget");
  if (a.te_host_pin != "auto") note("--te-host-pin");
  if (!a.control_files.empty() || !a.control_images.empty()) note("--control");
  if (!a.references.empty()) note("--reference");
  if (!a.init_image.empty() || !a.mask.empty()) note("--init-image/--mask");
  if (!a.hires.empty()) note("--hires");
  if (!a.distill.empty() || !a.time_embedding.empty()) note("--distill/--time-embedding");
  if (!a.dataspace.empty()) note("--dataspace");
  if (!a.preview.empty()) note("--preview");
  if (!a.then_prompts.empty()) note("--then-prompt");
  if (a.noise_source != "philox") note("--noise-source");
  if (a.eta != 0) note("--eta");
  if (a.s_noise != 1) note("--s-noise");
  if (a.slg_scale != 0 || !a.slg.empty()) note("--slg-*");
  if (a.cfg_truncation != 1.0) note("--cfg-truncation");
  if (a.list) return a;
  if (a.model.empty() || !a.have_prompt) usage("--model and --prompt are required");
  if (a.out_png.empty() && a.out_latent.empty()) usage("at least one of -o / --latent-out is required");
  if (a.device < -1) usage("--device takes a device index, or -1 for the first usable device");
  // pipeline::validate()'s checks and messages, for the settings this program takes.
  if (tok::python_strip(a.prompt).empty()) usage("Enter a prompt.");
  if (a.height <= 0 || a.width <= 0 || a.height % 16 != 0 || a.width % 16 != 0) {
    usage("Height and width must be positive multiples of 16.");
  }
  if (a.height > kMaxSide || a.width > kMaxSide) usage("Height and width must be at most " + std::to_string(kMaxSide) + ".");
  if (std::int64_t{a.height} * a.width > kMaxPixels) {
    usage("Image size must be at most 4.2 megapixels (e.g. 2048x2048); the model is meant for 0.25 to 4 megapixels.");
  }
  if (a.steps < 1) usage("Steps and image count must be positive.");
  if (a.steps > kMaxSteps) usage("Steps must be at most " + std::to_string(kMaxSteps) + ".");
  // These arrive as doubles but the sampler stores floats: a finite double above FLT_MAX would become inf here and
  // poison the latents, so reject it as a usage error rather than silently producing NaNs.
  const auto finite_f32 = [&](double v, const char* what) {
    if (!std::isfinite(v) || std::fabs(v) > static_cast<double>(std::numeric_limits<float>::max()))
      usage(std::string(what) + " must be finite and within the float range.");
  };
  finite_f32(a.cfg, "CFG");
  finite_f32(a.shift, "shift");
  finite_f32(a.slg_scale, "slg-scale");
  finite_f32(a.slg_start, "slg-start");
  finite_f32(a.slg_stop, "slg-stop");
  finite_f32(a.cfg_truncation, "cfg-truncation");
  finite_f32(a.eta, "eta");
  finite_f32(a.s_noise, "s-noise");
  if (a.cfg < 0 || a.shift <= 0) usage("CFG must be finite and nonnegative; shift must be finite and positive.");
  if (a.seed < -1 || a.seed >= kMaxSeed) usage("Seed must be -1 (random) or an integer from 0 to 2^53-1 (browser-safe range).");
  if (a.max_seq < 1 || a.max_seq > static_cast<int>(tok::kMossMaxSequenceLength)) {
    usage("Maximum text length must be between 1 and 4096.");
  }
  if (!std::isfinite(a.flow_multiplier) || a.flow_multiplier <= 0) usage("--flow-multiplier must be positive and finite.");
  if (a.eta < 0 || a.s_noise < 0) usage("--eta and --s-noise must be nonnegative.");
  if (a.cfg_truncation < 0 || a.cfg_truncation > 1) usage("--cfg-truncation must be between 0 and 1.");
  if (a.slg_scale < 0 || a.slg_start < 0 || a.slg_stop > 1 || a.slg_start >= a.slg_stop) {
    usage("--slg-scale must be nonnegative and 0 <= --slg-start < --slg-stop <= 1.");
  }
  if (a.strength <= 0 || a.strength > 1 || a.hires_strength <= 0 || a.hires_strength > 1) {
    usage("--strength and --hires-strength must be in (0, 1].");
  }
  if (a.direct_io && a.buffered_io) usage("--direct-io and --buffered-io contradict each other");
  return a;
}

int list_devices() {
  const std::vector<hip::DeviceInfo> devices = hip::list_devices();
  if (devices.empty()) std::printf("no HIP devices\n");
  for (const hip::DeviceInfo& d : devices) {
    std::printf("device %d: %s (%s), %.1f GiB\n", d.index, d.name.c_str(), d.arch.c_str(),
                static_cast<double>(d.total_memory) / static_cast<double>(1ull << 30));
    if (!d.unusable_reason.empty()) {
      std::printf("  unusable: %s\n", d.unusable_reason.c_str());
    } else {
      std::printf("  kernels: %s, %s\n", d.arch.c_str(), hip_block::leaf_name());
    }
  }
  return 0;
}

int run(const Args& a) {
  hip::Device device(a.device);
  // Always reported: which device and kernels ran should be visible (the NVIDIA build's scalar leaf is a logic check).
  std::fprintf(stderr, "hip: device %d, %s (%s); kernels: %s, %s\n", device.info().index, device.info().name.c_str(),
               device.info().arch.c_str(), hip::build_target(), hip_block::leaf_name());
  if (a.sampler != "euler") {
    throw std::runtime_error("moss-infer-hip: --sampler " + a.sampler +
                             " is not implemented by this backend; only euler is available");
  }

  for (const fs::path* p : {&a.dit, &a.te, &a.vae, &a.lora}) {
    if (!p->empty() && !fs::is_regular_file(*p)) {
      throw std::runtime_error("moss-infer-hip: file not found: " + util::path_utf8(*p));
    }
  }
  if (!a.lora_placement.empty() && a.lora_placement != "side_branch" && a.lora_placement != "merge" &&
      a.lora_placement != "default") {
    usage("--lora-placement takes default, merge or side_branch");
  }
  if (!a.lora.empty() && a.lora_placement != "side_branch") {
    std::fprintf(stderr, "moss-infer-hip: note: LoRA runs as a side branch (--lora-placement=%s)\n",
                 a.lora_placement.empty() ? "default" : a.lora_placement.c_str());
  }
  if (a.lora_strict) std::fprintf(stderr, "moss-infer-hip: note: --lora-strict is not enforced by this backend\n");

  const tok::Tokenizer tokenizer = tok::Tokenizer::load(a.model / "tokenizer" / "tokenizer.json");
  hip::GenerateRequest g;
  g.model_dir = a.model;
  g.dit_file = a.dit;
  g.te_file = a.te;
  g.vae_file = a.vae;
  g.lora_file = a.lora;
  g.lora_scale = a.lora_scale;
  tok::PromptTokenReport cond_report, uncond_report;
  g.cond_ids = tok::encode_moss_prompt_report(tokenizer, a.prompt, static_cast<std::size_t>(a.max_seq), cond_report);
  const bool cfg = a.cfg != 1.0;
  if (cfg) {
    g.uncond_ids = tok::encode_moss_prompt_report(tokenizer, a.negative, static_cast<std::size_t>(a.max_seq), uncond_report);
  }
  for (const auto* t : {&cond_report, &uncond_report}) {
    if (t->truncated) {
      std::fprintf(stderr, "warning: %s truncated to %zu of %zu tokens (%zu dropped; --max-seq %zu)\n",
                   t == &cond_report ? "prompt" : "negative prompt", t->kept_tokens, t->total_tokens, t->dropped_tokens(),
                   t->max_length);
    }
  }

  std::uint64_t seed = 0;
  if (a.seed >= 0) {
    seed = static_cast<std::uint64_t>(a.seed);
  } else {  // as the engine: a random 32-bit seed, reported
    std::random_device rd;
    seed = std::uniform_int_distribution<std::uint64_t>(0, (std::uint64_t{1} << 32) - 1)(rd);
  }
  g.width = static_cast<std::uint32_t>(a.width);
  g.height = static_cast<std::uint32_t>(a.height);
  g.noise = pipeline::initial_noise(pipeline::parse_noise_layout(a.noise_layout), seed, hip::kDiTLatentChannels,
                                    a.height / 16, a.width / 16);
  const pipeline::FlowSampling flow{a.shift, a.flow_multiplier};
  const pipeline::Schedule schedule = pipeline::make_schedule(pipeline::parse_scheduler(a.scheduler), a.steps, flow);
  g.sigmas = schedule.sigmas;
  g.timesteps = schedule.timesteps;
  g.cfg = static_cast<float>(a.cfg);
  for (const int l : a.skip) g.uncond_skip.set(static_cast<std::size_t>(l));
  g.vram_budget = a.vram_budget;
  g.int8_dit = (a.tier == "high");
  g.vae_tiled = a.vae_tiled;
  g.decode = !a.out_png.empty();
  if (!a.out_latent.empty()) {
    g.on_latents = [&](const std::vector<float>& latents) {
      hip::npy::save(a.out_latent, "<f4", {hip::kDiTLatentChannels, a.height / 16, a.width / 16}, latents.data(),
                     latents.size() * sizeof(float));
    };
  }
  double last = 0;
  if (!a.quiet) {
    g.progress = [&](const char* stage, int done, int total) {
      const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      if (std::string_view(stage) == "denoise" && done > 0) {
        // Like the CUDA CLI: redraw one line in place instead of one line per step.
        // Fixed-width time so a shorter value cannot leave stale digits from the previous step on the line.
        std::fprintf(stderr, "\rstep %d/%d  %7.2f s/step    %s", done, total, now - last, done == total ? "\n" : "");
      } else if (done == 0) {
        std::fprintf(stderr, "%s\n", stage);
      }
      last = now;
    };
  }

  hip::GenerateResult r;
  try {
    r = hip::generate(device, g);
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(e.what());  // an internal check, not a usage error: the request was checked above
  }
  if (!a.out_png.empty()) {
    std::vector<pipeline::PngText> text;
    if (a.metadata) {
      pipeline::GenerationMetadata m;
      m.engine_version = std::to_string(MI_VERSION_MAJOR) + "." + std::to_string(MI_VERSION_MINOR) + "." +
                         std::to_string(MI_VERSION_PATCH);
      m.tier = a.tier;
      m.models.push_back({"bundle", util::path_utf8(util::path_key(a.model).filename()), ""});
      m.prompt = a.prompt;
      m.negative_prompt = a.negative;
      m.sampler = "euler";
      m.scheduler = pipeline::scheduler_name(pipeline::parse_scheduler(a.scheduler));
      m.steps = a.steps;
      m.cfg = a.cfg;
      m.shift = a.shift;
      m.seed = seed;
      m.width = a.width;
      m.height = a.height;
      m.extra.emplace_back("noise_layout", pipeline::noise_layout_name(pipeline::parse_noise_layout(a.noise_layout)));
      m.extra.emplace_back("backend", "hip " + device.info().arch);
      text = pipeline::generation_text_chunks(m);
    }
    pipeline::write_png_rgb8(a.out_png, r.rgb, static_cast<int>(r.width), static_cast<int>(r.height), text);
  }
  if (!a.json_report.empty()) {
    const auto num = [](double x) { char b[32]; std::snprintf(b, sizeof b, "%.9g", x); return std::string(b); };
    const auto path = [](const fs::path& p) { return json_escape(util::path_utf8(p)); };
    std::string doc = "{\n";
    doc += "  \"ok\": true,\n";
    doc += "  \"backend\": " + json_escape("hip " + device.info().arch) + ",\n";
    doc += "  \"models\": {\"bundle\": " + path(a.model.filename()) +
           (a.dit.empty() ? "" : ", \"dit\": " + path(a.dit.filename())) +
           (a.te.empty() ? "" : ", \"te\": " + path(a.te.filename())) +
           (a.vae.empty() ? "" : ", \"vae\": " + path(a.vae.filename())) +
           (a.lora.empty() ? "" : ", \"lora\": " + path(a.lora.filename())) + "},\n";
    doc += "  \"prompt\": " + json_escape(a.prompt) + ",\n";
    doc += "  \"negative_prompt\": " + json_escape(a.negative) + ",\n";
    doc += "  \"width\": " + std::to_string(r.width) + ", \"height\": " + std::to_string(r.height) + ",\n";
    doc += "  \"steps\": " + std::to_string(a.steps) + ", \"cfg\": " + num(a.cfg) + ", \"shift\": " + num(a.shift) +
           ", \"seed\": " + std::to_string(seed) + ",\n";
    doc += "  \"sampler\": \"euler\", \"scheduler\": " + json_escape(a.scheduler) + ", \"tier\": " +
           json_escape(a.tier) + ",\n";
    doc += "  \"lora_scale\": " + num(a.lora_scale) + ", \"lora_placement\": " +
           json_escape(a.lora_placement.empty() ? "side_branch" : a.lora_placement) + ",\n";
    doc += "  \"timings_s\": {\"text_encoder_load\": " + num(r.text_encoder_load_s) +
           ", \"text_encoder\": " + num(r.text_encoder_s) + ", \"dit_load\": " + num(r.dit_load_s) +
           ", \"denoise\": " + num(r.denoise_s) + ", \"vae\": " + num(r.vae_s) + "},\n";
    doc += "  \"dit_resident_blocks\": " + std::to_string(r.dit_resident_blocks) + ", \"vae_decode\": " +
           json_escape(vae_decode_label(r)) + "\n}\n";
    std::ofstream out(a.json_report, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + util::path_utf8(a.json_report));
    out << doc;
  }
  if (!a.quiet) {
    const std::string decode = vae_decode_label(r);
    std::fprintf(stderr, "done: seed %llu; text encoder load %.1f s, encode %.1f s, DiT load %.1f s (%d of 42 blocks on the "
                         "device, %.2f GiB free beside them), denoise %.1f s (%.2f s/step), VAE %.1f s (%s)\n",
                 static_cast<unsigned long long>(seed), r.text_encoder_load_s, r.text_encoder_s, r.dit_load_s,
                 r.dit_resident_blocks,
                 static_cast<double>(r.dit_free_bytes) / static_cast<double>(1ull << 30), r.denoise_s,
                 r.denoise_s / static_cast<double>(g.timesteps.size()), r.vae_s, decode.c_str());
  }
  return 0;
}

int run_main(const std::vector<std::string>& argv) {
  Args a;
  try {
    a = parse(argv);
  } catch (const std::invalid_argument& e) {
    if (e.what()[0] != '\0') std::fprintf(stderr, "moss-infer-hip: %s\n", e.what());
    std::fputs(kUsage, stderr);
    return 2;
  } catch (const std::exception& e) {
    // --params reads and parses a file: a missing or malformed file throws std::runtime_error, not
    // std::invalid_argument. Without this handler the exception escapes run_main() and terminates the process.
    std::fprintf(stderr, "moss-infer-hip: %s\n", e.what());
    std::fputs(kUsage, stderr);
    return 2;
  }
  if (a.help) {
    std::fputs(kUsage, stdout);
    return 0;
  }
  if (a.version) {
    std::printf("moss-infer-hip %d.%d.%d (kernels: %s)\n", MI_VERSION_MAJOR, MI_VERSION_MINOR, MI_VERSION_PATCH,
                hip::build_target());
    return 0;
  }
  try {
    return a.list ? list_devices() : run(a);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "moss-infer-hip: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "moss-infer-hip: %s\n", e.what());
    return 1;
  }
}

}  // namespace

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// char** argv is encoded in the process' ANSI code page on Windows, so characters outside that page (e.g. Chinese in
// --prompt) are lost. Take the UTF-16 command line and convert every argument to UTF-8, as the CUDA CLI's entry point
// does; the tokenizer and the JSON writer both expect UTF-8.
int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> utf8;
  utf8.reserve(static_cast<std::size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? static_cast<std::size_t>(n) : 1, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, s.data(), n, nullptr, nullptr);
    s.pop_back();  // the converted terminator
    utf8.push_back(std::move(s));
  }
  return run_main(utf8);
}
#else
int main(int argc, char** argv) { return run_main(std::vector<std::string>(argv, argv + argc)); }
#endif
