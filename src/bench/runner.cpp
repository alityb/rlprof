#include "rlprof/bench/runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace rlprof::bench {
namespace {

class ChronoBackend final : public BenchmarkBackend {
 public:
  double measure_ms(const std::function<void()>& fn) override {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }
};

double percentile(std::vector<double> values, double quantile) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil((values.size() - 1) * quantile));
  return values[index];
}

std::size_t dtype_size(const std::string& dtype) {
  if (dtype == "bf16" || dtype == "fp16") {
    return 2;
  }
  if (dtype == "fp32") {
    return 4;
  }
  throw std::runtime_error("unsupported dtype: " + dtype);
}

}  // namespace

std::vector<Shape> parse_shapes(const std::string& spec) {
  std::vector<Shape> shapes;
  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    Shape shape;
    std::stringstream shape_stream(token);
    std::string dim;
    while (std::getline(shape_stream, dim, 'x')) {
      const std::int64_t value = std::stoll(dim);
      if (value <= 0) {
        throw std::runtime_error("invalid shape: " + token);
      }
      shape.push_back(value);
    }
    if (shape.empty()) {
      throw std::runtime_error("invalid shape: " + token);
    }
    shapes.push_back(shape);
  }
  if (shapes.empty()) {
    throw std::runtime_error("at least one shape is required");
  }
  return shapes;
}

std::vector<BenchResult> benchmark_impl(
    const std::string& category,
    const KernelImpl& implementation,
    const std::vector<Shape>& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    BenchmarkBackend* backend) {
  if (warmup < 0 || n_iter <= 0) {
    throw std::runtime_error("warmup must be >= 0 and n_iter must be > 0");
  }
  if (std::find(implementation.dtypes.begin(), implementation.dtypes.end(), dtype) ==
      implementation.dtypes.end()) {
    throw std::runtime_error("unsupported dtype for implementation: " + dtype);
  }

  ChronoBackend default_backend;
  BenchmarkBackend* active_backend = backend == nullptr ? &default_backend : backend;

  std::vector<BenchResult> results;
  for (const Shape& shape : shapes) {
    std::any state = implementation.setup(shape, dtype);
    for (std::int64_t i = 0; i < warmup; ++i) {
      implementation.fn(state);
    }
    active_backend->synchronize();

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(n_iter));
    for (std::int64_t i = 0; i < n_iter; ++i) {
      times.push_back(active_backend->measure_ms([&]() { implementation.fn(state); }));
      active_backend->synchronize();
    }

    const double avg_ms =
        std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    const double min_ms = *std::min_element(times.begin(), times.end());
    const double p50_ms = percentile(times, 0.50);
    const double p99_ms = percentile(times, 0.99);
    const double bandwidth_gb_s =
        static_cast<double>(implementation.bytes_processed(shape, dtype)) / (avg_ms / 1000.0) / 1e9;

    results.push_back(BenchResult{
        .kernel = category,
        .implementation = implementation.name,
        .shape = shape,
        .dtype = dtype,
        .avg_ms = avg_ms,
        .min_ms = min_ms,
        .p50_ms = p50_ms,
        .p99_ms = p99_ms,
        .bandwidth_gb_s = bandwidth_gb_s,
    });
  }
  return results;
}

std::vector<BenchResult> benchmark_category(
    const std::string& category,
    const std::vector<Shape>& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    BenchmarkBackend* backend) {
  std::vector<BenchResult> results;
  for (const KernelImpl& implementation : get_kernel_impls(category)) {
    const auto impl_results =
        benchmark_impl(category, implementation, shapes, dtype, warmup, n_iter, backend);
    results.insert(results.end(), impl_results.begin(), impl_results.end());
  }
  return results;
}

std::string render_bench_results(const std::vector<BenchResult>& results) {
  std::ostringstream out;
  out << std::left << std::setw(18) << "kernel" << "  "
      << std::setw(18) << "implementation" << "  "
      << std::setw(12) << "shape" << "  "
      << std::right << std::setw(8) << "avg ms" << "  "
      << std::setw(8) << "min ms" << "  "
      << std::setw(8) << "p50 ms" << "  "
      << std::setw(8) << "p99 ms" << "  "
      << std::setw(11) << "GB/s" << "\n";
  out << std::string(105, '-') << "\n";
  for (const BenchResult& result : results) {
    std::ostringstream shape_stream;
    for (std::size_t i = 0; i < result.shape.size(); ++i) {
      if (i > 0) {
        shape_stream << "x";
      }
      shape_stream << result.shape[i];
    }
    out << std::left << std::setw(18) << result.kernel << "  "
        << std::setw(18) << result.implementation << "  "
        << std::setw(12) << shape_stream.str() << "  "
        << std::right << std::fixed << std::setprecision(3)
        << std::setw(8) << result.avg_ms << "  "
        << std::setw(8) << result.min_ms << "  "
        << std::setw(8) << result.p50_ms << "  "
        << std::setw(8) << result.p99_ms << "  "
        << std::setw(11) << result.bandwidth_gb_s << "\n";
  }
  return out.str();
}

}  // namespace rlprof::bench
