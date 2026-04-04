#pragma once

#include <cstdint>
#include <string_view>

namespace hotpath::profiler {

std::string_view categorize(std::string_view kernel_name);

enum class KernelPhase {
  PREFILL,
  DECODE,
  UNKNOWN,
};

struct GridDim {
  int x = 1;
  int y = 1;
  int z = 1;
};

KernelPhase classify_phase(std::string_view kernel_name,
                           GridDim grid = {},
                           int64_t grid_threshold = 1024);

}  // namespace hotpath::profiler
