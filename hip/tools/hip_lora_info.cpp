// Print the LoRA side-branch shapes the HIP backend actually executes.
#include <cstdio>

#include "lora.h"

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  mi::hip::Device device(-1);
  mi::hip::Lora lora(argv[1], 1.0f, 42);
  std::printf("adapters=%d max_rank=%d max_n=%d\n", lora.adapters(), lora.max_rank(), lora.max_n());
  for (int l = 0; l < 3; ++l) {
    const auto& b = lora.block(l);
    for (int s = 0; s < mi::hip::kLoraSites; ++s) {
      const auto& br = b.site[s];
      std::printf("block %d site %d: active=%d m=6208 n=%d k=%d rank=%d\n", l, s, br.active, br.n, br.k, br.rank);
    }
  }
  return 0;
}
