#include "rlprof/profiler/categorizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

namespace rlprof::profiler {
namespace {

struct PatternGroup {
  std::string_view category;
  std::array<std::string_view, 10> patterns;
  std::size_t count;
};

constexpr std::array<PatternGroup, 11> kPatternGroups = {{
    {"gemm", {"xmma_gemm", "cutlass", "cublas", "nvjet", "wgmma", "gemm", "ampere_bf16", "hmma", "mma", ""}, 9},
    {"attention", {"flash", "fmha", "paged_attention", "flashattnfwd", "softmax", "splitkv", "attention", "", "", ""}, 7},
    {"moe", {"fused_moe", "expert", "topk_softmax", "moe_align", "", "", "", "", "", ""}, 4},
    {"mamba", {"mamba", "selective_scan", "ssm", "", "", "", "", "", "", ""}, 3},
    {"activation", {"silu", "gelu", "relu", "act_and_mul", "swiglu", "elementwise", "fused_mul", "", "", ""}, 7},
    {"norm", {"rms_norm", "layer_norm", "fused_add_rms", "pow_rsqrt", "mean_mul", "rsqrt", "", "", "", ""}, 6},
    {"position", {"rotary", "rope", "", "", "", "", "", "", "", ""}, 2},
    {"cache", {"reshape_and_cache", "cache_kernel", "kv_cache", "", "", "", "", "", "", ""}, 3},
    {"sampling", {"top_k", "top_p", "sample", "argmax", "topk_topp", "topp", "", "", "", ""}, 6},
    {"comm", {"allreduce", "allgather", "nccl", "reduce_scatter", "", "", "", "", "", ""}, 4},
    {"memory", {"memcpy", "memset", "fill_kernel", "copy", "", "", "", "", "", ""}, 4},
}};

std::string to_lower(std::string_view input) {
  std::string lowered;
  lowered.reserve(input.size());
  for (unsigned char ch : input) {
    lowered.push_back(static_cast<char>(std::tolower(ch)));
  }
  return lowered;
}

}  // namespace

std::string_view categorize(std::string_view kernel_name) {
  const std::string lowered_name = to_lower(kernel_name);

  for (const PatternGroup& group : kPatternGroups) {
    for (std::size_t i = 0; i < group.count; ++i) {
      const std::string lowered_pattern = to_lower(group.patterns[i]);
      if (lowered_name.find(lowered_pattern) != std::string::npos) {
        return group.category;
      }
    }
  }

  return "other";
}

}  // namespace rlprof::profiler
