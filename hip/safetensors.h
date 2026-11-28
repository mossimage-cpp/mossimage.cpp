// A minimal safetensors reader: the header of one file, tensor bytes read with pread, and the shard map of a
// model.safetensors.index.json. Failures throw std::runtime_error.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mi::hip::st {

struct TensorInfo {
  std::string name;
  std::string dtype;  // as stored, e.g. "BF16", "F32"
  std::vector<std::int64_t> shape;
  std::uint64_t offset = 0;  // of the first byte, from the start of the file
  std::uint64_t bytes = 0;
};

class File {
 public:
  explicit File(const std::filesystem::path& path);  // opened by its wide name on Windows
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  const std::string& path() const { return path_; }  // UTF-8, for messages
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  // File-level `__metadata__` value, or null when the key is absent.
  const std::string* metadata(const std::string& key) const;
  const TensorInfo* find(const std::string& name) const;  // null if absent
  const TensorInfo& at(const std::string& name) const;    // throws if absent
  // lora_adapter_metadata's lora_alpha, as PEFT/diffusers LoRA files store it when there is no alpha tensor.
  std::optional<double> lora_metadata_alpha() const;

  // `bytes` bytes of tensor `t` from byte `rel` of it into `dst`.
  void read(const TensorInfo& t, void* dst, std::uint64_t rel, std::uint64_t bytes) const;
  void read(const TensorInfo& t, void* dst) const { read(t, dst, 0, t.bytes); }
  // A whole F32 or BF16 tensor as fp32 values (bf16 widened exactly), or throws.
  std::vector<float> read_f32(const std::string& name) const;

 private:
  std::string path_;
  int fd_ = -1;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::size_t> by_name_;
  std::map<std::string, std::string> metadata_;
};

// model.safetensors.index.json's weight_map: tensor name -> shard file name (relative to the index's directory).
std::map<std::string, std::string> read_index(const std::filesystem::path& path);

// `_quantization_metadata`'s layers: canonical module name -> ConvRot group of the modules stored as INT8 ConvRot.
// Other formats and unrotated modules are left out; the caller validates the group before using it.
std::map<std::string, int> convrot_groups(const std::string& json);

// The file holding every tensor name, for one .safetensors file, a directory with a weight_map index, or a directory
// with exactly one .safetensors. Unlike Index, this keeps the file paths so a caller can open them itself.
std::map<std::string, std::filesystem::path> index_files(const std::filesystem::path& path);

// The tensors of one file, or of every shard of a model directory, by name. DiT tensor names are canonicalized:
// a `model.diffusion_model.` / `diffusion_model.` prefix is stripped, as the CUDA backend does.
class Index {
 public:
  // A .safetensors file, or a directory holding one *.safetensors.index.json (its shards) or exactly one .safetensors.
  explicit Index(const std::filesystem::path& path);

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  // File-level `__metadata__` value from any of the shards, or null when absent.
  const std::string* metadata(const std::string& key) const;
  const TensorInfo& at(const std::string& name) const;  // throws if absent
  // Reads a whole tensor into caller memory of exactly its size, or throws.
  void read_into(const std::string& name, void* out, std::uint64_t bytes) const;

 private:
  std::vector<std::unique_ptr<File>> files_;
  std::map<std::string, std::pair<const File*, const TensorInfo*>> tensors_;
};

}  // namespace mi::hip::st
