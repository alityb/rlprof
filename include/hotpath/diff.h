#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "hotpath/profiler/kernel_record.h"

namespace hotpath {

struct CategoryDelta {
  std::string category;
  double a_ms;
  double b_ms;
  double delta_ms;
  std::optional<double> delta_pct;
};

std::vector<CategoryDelta> diff_kernel_categories(
    const std::vector<profiler::KernelRecord>& kernels_a,
    const std::vector<profiler::KernelRecord>& kernels_b);

std::vector<CategoryDelta> diff_profiles(
    const std::filesystem::path& path_a,
    const std::filesystem::path& path_b);

std::string render_diff(
    const std::filesystem::path& path_a,
    const std::filesystem::path& path_b,
    bool color = false);

}  // namespace hotpath
