#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "rlprof/aggregate.h"
#include "rlprof/store.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_test_aggregate";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root);

  rlprof::ProfileData left;
  left.meta = {{"model_name", "Qwen/Qwen3-8B"}, {"gpu_name", "NVIDIA A10G"}};
  left.kernels = {{
      .name = "flash_fwd_splitkv_kernel",
      .category = "attention",
      .total_ns = 100,
      .calls = 1,
      .avg_ns = 100,
      .min_ns = 100,
      .max_ns = 100,
      .registers = 32,
      .shared_mem = 64,
  }};
  left.metrics = {{.sample_time = 0.0, .source = "node0", .metric = "vllm:num_requests_running", .value = 2.0}};
  left.traffic_stats = {.total_requests = 4, .completion_length_mean = 10.0, .completion_length_p50 = 8.0, .completion_length_p99 = 12.0, .max_median_ratio = 1.5, .errors = 1};
  const fs::path left_path = temp_root / "left.db";
  rlprof::save_profile(left_path, left);

  rlprof::ProfileData right = left;
  right.kernels[0].total_ns = 200;
  right.kernels[0].calls = 2;
  right.kernels[0].avg_ns = 100;
  right.metrics[0].source = "node1";
  right.metrics[0].value = 4.0;
  right.traffic_stats.total_requests = 6;
  right.traffic_stats.completion_length_mean = 20.0;
  right.traffic_stats.completion_length_p99 = 24.0;
  right.traffic_stats.errors = 0;
  const fs::path right_path = temp_root / "right.db";
  rlprof::save_profile(right_path, right);

  const auto aggregate = rlprof::aggregate_profiles({left_path, right_path});
  expect_true(aggregate.kernels.size() == 1, "expected merged kernel");
  expect_true(aggregate.kernels[0].total_ns == 300, "expected summed kernel total");
  expect_true(aggregate.kernels[0].calls == 3, "expected summed kernel calls");
  expect_true(aggregate.metrics.size() == 2, "expected merged metrics");
  expect_true(aggregate.metrics_summary.size() == 1, "expected recomputed metric summaries");
  expect_true(aggregate.traffic_stats.total_requests == 10, "expected total requests sum");
  expect_true(aggregate.traffic_stats.errors == 1, "expected total error sum");
  expect_true(
      aggregate.traffic_stats.completion_length_mean.has_value() &&
          *aggregate.traffic_stats.completion_length_mean == 16.0,
      "expected weighted completion mean");

  fs::remove_all(temp_root);
  return 0;
}
