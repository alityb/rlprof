#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rlprof/bench/registry.h"

namespace rlprof::bench {

struct BenchResult {
  std::string kernel;
  std::string implementation;
  Shape shape;
  std::string dtype;
  double avg_ms;
  double min_ms;
  double p50_ms;
  double p99_ms;
  double bandwidth_gb_s;
};

class BenchmarkBackend {
 public:
  virtual ~BenchmarkBackend() = default;
  virtual double measure_ms(const std::function<void()>& fn) = 0;
  virtual void synchronize() {}
};

std::vector<Shape> parse_shapes(const std::string& spec);

std::vector<BenchResult> benchmark_impl(
    const std::string& category,
    const KernelImpl& implementation,
    const std::vector<Shape>& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    BenchmarkBackend* backend = nullptr);

std::vector<BenchResult> benchmark_category(
    const std::string& category,
    const std::vector<Shape>& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    BenchmarkBackend* backend = nullptr);

std::string render_bench_results(const std::vector<BenchResult>& results);

}  // namespace rlprof::bench
