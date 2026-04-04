#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/batch_analyzer.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  // 60 synthetic snapshots: batch size ramps 1→32 then back to 1
  std::vector<hotpath::MetricSnapshot> snapshots;
  snapshots.reserve(60);

  for (int i = 0; i < 60; ++i) {
    hotpath::MetricSnapshot s;
    s.timestamp_us = static_cast<int64_t>(i) * 1000000;
    // Ramp up then down
    if (i < 30) {
      s.batch_size = 1.0 + (31.0 * i / 29.0);
    } else {
      s.batch_size = 1.0 + (31.0 * (59 - i) / 29.0);
    }
    s.queue_depth = (i < 30) ? static_cast<double>(i) : static_cast<double>(59 - i);
    s.preemption_total = static_cast<double>(i < 30 ? i / 5 : 6 + (i - 30) / 10);
    s.cache_usage = 50.0 + 40.0 * std::sin(static_cast<double>(i) * 0.1);
    snapshots.push_back(s);
  }

  const auto result = hotpath::analyze_batches(snapshots);

  // Verify averages are reasonable
  expect_true(result.avg_batch_size > 1.0 && result.avg_batch_size < 33.0,
              "avg_batch_size out of range: " + std::to_string(result.avg_batch_size));

  // p50 should be somewhere in the middle
  expect_true(result.p50_batch_size > 5.0 && result.p50_batch_size < 30.0,
              "p50_batch_size out of range: " + std::to_string(result.p50_batch_size));

  // p99 should be near the peak
  expect_true(result.p99_batch_size > 25.0,
              "p99_batch_size should be near peak: " + std::to_string(result.p99_batch_size));

  expect_true(result.avg_queue_depth >= 0.0, "avg_queue_depth should be >= 0");
  expect_true(result.p99_queue_depth >= 0.0, "p99_queue_depth should be >= 0");

  expect_true(result.total_preemptions >= 0, "total_preemptions should be >= 0");

  expect_true(result.avg_cache_usage > 0.0, "avg_cache_usage should be > 0");
  expect_true(result.peak_cache_usage >= result.avg_cache_usage,
              "peak should be >= avg cache usage");

  // Series length should match input
  expect_true(result.batch_size_series.size() == 60,
              "batch_size_series length: " + std::to_string(result.batch_size_series.size()));
  expect_true(result.queue_depth_series.size() == 60,
              "queue_depth_series length: " + std::to_string(result.queue_depth_series.size()));

  // Test empty input
  const auto empty = hotpath::analyze_batches({});
  expect_true(empty.avg_batch_size == 0.0, "empty should have 0 avg_batch_size");

  std::cerr << "test_batch_analyzer: all tests passed\n";
  return 0;
}
