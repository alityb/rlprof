#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "hotpath/phase_analyzer.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  using hotpath::KernelEntry;
  using hotpath::profiler::KernelPhase;

  // Create 200 synthetic kernel entries over 5 seconds
  std::vector<KernelEntry> kernels;
  kernels.reserve(200);

  int64_t expected_prefill_us = 0;
  int64_t expected_decode_us = 0;
  int64_t expected_unknown_us = 0;
  int prefill_count = 0;
  int decode_count = 0;

  for (int i = 0; i < 200; ++i) {
    KernelEntry k;
    k.name = "kernel_" + std::to_string(i);
    k.start_us = static_cast<int64_t>(i) * 25000;  // 25ms apart, spanning 5s
    k.duration_us = 10000;  // 10ms each

    if (i % 3 == 0) {
      k.phase = KernelPhase::PREFILL;
      expected_prefill_us += k.duration_us;
      prefill_count++;
    } else if (i % 3 == 1) {
      k.phase = KernelPhase::DECODE;
      expected_decode_us += k.duration_us;
      decode_count++;
    } else {
      k.phase = KernelPhase::UNKNOWN;
      expected_unknown_us += k.duration_us;
    }
    kernels.push_back(k);
  }

  const auto result = hotpath::analyze_phases(kernels);
  const auto& bd = result.breakdown;

  // Verify totals
  expect_true(bd.prefill_us == expected_prefill_us,
              "prefill_us mismatch: " + std::to_string(bd.prefill_us) +
              " vs " + std::to_string(expected_prefill_us));
  expect_true(bd.decode_us == expected_decode_us,
              "decode_us mismatch: " + std::to_string(bd.decode_us) +
              " vs " + std::to_string(expected_decode_us));
  expect_true(bd.unknown_us == expected_unknown_us, "unknown_us mismatch");
  expect_true(bd.total_us == expected_prefill_us + expected_decode_us + expected_unknown_us,
              "total_us mismatch");

  // Verify counts
  expect_true(bd.prefill_kernel_count == prefill_count,
              "prefill count: " + std::to_string(bd.prefill_kernel_count));
  expect_true(bd.decode_kernel_count == decode_count,
              "decode count: " + std::to_string(bd.decode_kernel_count));

  // Verify fractions in [0, 1]
  expect_true(bd.prefill_fraction >= 0.0 && bd.prefill_fraction <= 1.0,
              "prefill_fraction out of range");
  expect_true(bd.decode_fraction >= 0.0 && bd.decode_fraction <= 1.0,
              "decode_fraction out of range");
  expect_true(bd.prefill_fraction + bd.decode_fraction <= 1.0 + 1e-9,
              "fractions sum > 1");

  // Verify time series
  // Kernels span from 0 to 199*25000 + 10000 = 4985000 us ≈ 5 seconds
  // So we should have 5 windows
  expect_true(result.time_series.size() == 5,
              "expected 5 time series windows, got " +
              std::to_string(result.time_series.size()));

  for (const auto& tp : result.time_series) {
    expect_true(tp.prefill_fraction >= 0.0 && tp.prefill_fraction <= 1.0,
                "time series prefill fraction out of range");
    expect_true(tp.decode_fraction >= 0.0 && tp.decode_fraction <= 1.0,
                "time series decode fraction out of range");
  }

  // Test empty input
  const auto empty_result = hotpath::analyze_phases({});
  expect_true(empty_result.breakdown.total_us == 0, "empty should have 0 total");
  expect_true(empty_result.time_series.empty(), "empty should have no time series");

  std::cerr << "test_phase_analyzer: all tests passed\n";
  return 0;
}
