#include "hotpath/diff.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>

#include <sys/ioctl.h>
#include <unistd.h>

#include "hotpath/store.h"

namespace hotpath {
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

std::string rule(int width) {
  std::string result;
  result.reserve(static_cast<std::size_t>(width) * 3);
  for (int i = 0; i < width; ++i) {
    result += "\xe2\x94\x80";  // ─
  }
  return result;
}

int detect_terminal_width() {
  struct winsize ws{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return std::max(static_cast<int>(ws.ws_col), 80);
  }
  return 80;
}

std::string render_bar(double percent, int max_width) {
  const double abs_pct = std::fabs(percent);
  const double capped = std::min(abs_pct, 100.0);
  const double filled = (capped / 100.0) * max_width;
  const int full_blocks = std::min(static_cast<int>(filled), max_width);
  const double remainder = filled - full_blocks;
  std::string bar;
  for (int i = 0; i < full_blocks; ++i) {
    bar += "\xe2\x96\x88";  // █
  }
  if (remainder >= 0.5 && full_blocks < max_width) {
    bar += "\xe2\x96\x8c";  // ▌
  }
  return bar;
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
    const std::filesystem::path& path_b,
    bool color) {
  auto deltas = diff_profiles(path_a, path_b);

  // Sort by absolute delta (largest change first) for scannability
  std::sort(deltas.begin(), deltas.end(),
            [](const CategoryDelta& a, const CategoryDelta& b) {
              return std::fabs(a.delta_ms) > std::fabs(b.delta_ms);
            });

  const char* RST = color ? "\033[0m" : "";
  const char* BLD = color ? "\033[1m" : "";
  const char* DIM = color ? "\033[2m" : "";
  const char* RED = color ? "\033[31m" : "";
  const char* GRN = color ? "\033[32m" : "";
  const char* CYN = color ? "\033[36m" : "";

  const int term_width = color ? detect_terminal_width() : 80;
  constexpr int kFixedCols = 60;  // category(16) + A(10) + B(10) + delta(12) + pct(10) + gaps
  const int bar_width = std::clamp(term_width - kFixedCols - 4, 8, 24);

  std::ostringstream out;
  out << "\n" << BLD << CYN
      << "\xe2\x94\x80\xe2\x94\x80 CATEGORY DIFF "
      << RST << DIM
      << rule(term_width - 17) << RST << "\n";

  // File labels
  out << DIM << "  A: " << RST << path_a.filename().string() << "\n";
  out << DIM << "  B: " << RST << path_b.filename().string() << "\n\n";

  // Column headers (dim chrome)
  out << DIM
      << "  " << std::left << std::setw(16) << "category"
      << std::right << std::setw(10) << "A (ms)"
      << std::setw(10) << "B (ms)"
      << std::setw(12) << "\xce\x94 (ms)"  // Δ
      << std::setw(10) << "\xce\x94 (%)"
      << RST << "\n";
  out << "  " << DIM << rule(term_width - 4) << RST << "\n";

  // Find max absolute delta for bar scaling
  double max_abs_delta = 0.0;
  for (const CategoryDelta& delta : deltas) {
    max_abs_delta = std::max(max_abs_delta, std::fabs(delta.delta_ms));
  }

  // Compute totals for a summary row
  double total_a = 0.0;
  double total_b = 0.0;
  for (const CategoryDelta& delta : deltas) {
    total_a += delta.a_ms;
    total_b += delta.b_ms;
  }

  for (const CategoryDelta& delta : deltas) {
    const bool regression = delta.delta_ms > 0.001;
    const bool improvement = delta.delta_ms < -0.001;
    const char* delta_color = regression ? RED : (improvement ? GRN : DIM);

    // Arrow indicator
    const char* arrow = regression ? " \xe2\x86\x91" : (improvement ? " \xe2\x86\x93" : "  ");

    // Delta bar proportional to max absolute delta
    const double bar_pct = max_abs_delta > 0.0
        ? (std::fabs(delta.delta_ms) / max_abs_delta) * 100.0
        : 0.0;
    const std::string bar = render_bar(bar_pct, bar_width);

    // Format delta with sign
    std::string delta_str = (delta.delta_ms >= 0.0 ? "+" : "") + format_fixed(delta.delta_ms, 1);
    std::string pct_str = delta.delta_pct.has_value()
        ? ((delta.delta_ms >= 0.0 ? "+" : "") + format_fixed(*delta.delta_pct, 1) + "%")
        : "-";

    out << "  " << std::left << std::setw(16) << delta.category
        << std::right << std::setw(10) << format_fixed(delta.a_ms, 1)
        << std::setw(10) << format_fixed(delta.b_ms, 1)
        << delta_color << std::setw(10) << delta_str << arrow << RST
        << delta_color << std::setw(10) << pct_str << RST
        << "  " << delta_color << bar << RST << "\n";
  }

  // Summary total row
  const double total_delta = total_b - total_a;
  const bool total_regression = total_delta > 0.001;
  const bool total_improvement = total_delta < -0.001;
  const char* total_color = total_regression ? RED : (total_improvement ? GRN : DIM);
  const std::string total_delta_str = (total_delta >= 0.0 ? "+" : "") + format_fixed(total_delta, 1);
  const std::string total_pct_str = total_a > 0.0
      ? ((total_delta >= 0.0 ? "+" : "") + format_fixed((total_delta / total_a) * 100.0, 1) + "%")
      : "-";
  const char* total_arrow = total_regression ? " \xe2\x86\x91" : (total_improvement ? " \xe2\x86\x93" : "  ");

  out << "  " << DIM << rule(term_width - 4) << RST << "\n";
  out << "  " << BLD << std::left << std::setw(16) << "total" << RST
      << std::right << std::setw(10) << format_fixed(total_a, 1)
      << std::setw(10) << format_fixed(total_b, 1)
      << total_color << BLD << std::setw(10) << total_delta_str << total_arrow << RST
      << total_color << BLD << std::setw(10) << total_pct_str << RST << "\n";

  return out.str();
}

}  // namespace hotpath
