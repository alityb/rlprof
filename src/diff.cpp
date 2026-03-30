#include "rlprof/diff.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>

#include "rlprof/store.h"

namespace rlprof {
namespace {

std::map<std::string, std::int64_t> category_totals(
    const std::vector<profiler::KernelRecord>& kernels) {
  std::map<std::string, std::int64_t> totals;
  for (const profiler::KernelRecord& kernel : kernels) {
    totals[kernel.category] += kernel.total_ns;
  }
  return totals;
}

std::string format_fixed(double value, int precision) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

}  // namespace

std::vector<CategoryDelta> diff_kernel_categories(
    const std::vector<profiler::KernelRecord>& kernels_a,
    const std::vector<profiler::KernelRecord>& kernels_b) {
  const auto totals_a = category_totals(kernels_a);
  const auto totals_b = category_totals(kernels_b);

  std::map<std::string, bool> categories;
  for (const auto& [category, _] : totals_a) {
    categories[category] = true;
  }
  for (const auto& [category, _] : totals_b) {
    categories[category] = true;
  }

  std::vector<CategoryDelta> deltas;
  for (const auto& [category, _] : categories) {
    const double a_ms = static_cast<double>(totals_a.contains(category) ? totals_a.at(category) : 0) / 1'000'000.0;
    const double b_ms = static_cast<double>(totals_b.contains(category) ? totals_b.at(category) : 0) / 1'000'000.0;
    const double delta_ms = b_ms - a_ms;
    deltas.push_back(CategoryDelta{
        .category = category,
        .a_ms = a_ms,
        .b_ms = b_ms,
        .delta_ms = delta_ms,
        .delta_pct = a_ms == 0.0 ? std::nullopt : std::optional<double>((delta_ms / a_ms) * 100.0),
    });
  }
  return deltas;
}

std::vector<CategoryDelta> diff_profiles(
    const std::filesystem::path& path_a,
    const std::filesystem::path& path_b) {
  const ProfileData profile_a = load_profile(path_a);
  const ProfileData profile_b = load_profile(path_b);
  return diff_kernel_categories(profile_a.kernels, profile_b.kernels);
}

std::string render_diff(
    const std::filesystem::path& path_a,
    const std::filesystem::path& path_b) {
  const auto deltas = diff_profiles(path_a, path_b);

  std::ostringstream out;
  out << "CATEGORY DIFF\n";
  out << std::left << std::setw(16) << "category" << "  "
      << std::right << std::setw(8) << "A (ms)" << "  "
      << std::setw(8) << "B (ms)" << "  "
      << std::setw(10) << "delta (ms)" << "  "
      << std::setw(9) << "delta (%)" << "\n";
  out << std::string(59, '-') << "\n";
  for (const CategoryDelta& delta : deltas) {
    out << std::left << std::setw(16) << delta.category << "  "
        << std::right << std::setw(8) << format_fixed(delta.a_ms, 1) << "  "
        << std::setw(8) << format_fixed(delta.b_ms, 1) << "  "
        << std::setw(10) << format_fixed(delta.delta_ms, 1) << "  "
        << std::setw(9) << (delta.delta_pct.has_value() ? format_fixed(*delta.delta_pct, 1) : "-")
        << "\n";
  }
  return out.str();
}

}  // namespace rlprof
