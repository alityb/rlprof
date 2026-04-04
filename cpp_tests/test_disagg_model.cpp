#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/disagg_model.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  // Test 1: Short-context workload → should NOT disaggregate
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 256;
    input.profile.median_output_tokens = 64;
    input.profile.request_rate = 50.0;
    input.profile.prefill_fraction = 0.1;
    input.profile.prefill_contention = 0.2;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;

    auto est = hotpath::estimate_disaggregation(input);
    expect_true(!est.should_disaggregate,
                "short-context should NOT disaggregate: " + est.reason);
  }

  // Test 2: Long-context workload → should disaggregate
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 10.0;
    input.profile.prefill_fraction = 0.6;
    input.profile.prefill_contention = 1.5;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;

    auto est = hotpath::estimate_disaggregation(input);
    expect_true(est.should_disaggregate,
                "long-context should disaggregate: " + est.reason);
    // Optimal split should favor decode (more GPUs for decode since decode takes longer)
    expect_true(est.optimal_prefill_gpus >= 1 && est.optimal_prefill_gpus <= 4,
                "optimal_prefill_gpus: " + std::to_string(est.optimal_prefill_gpus));
    expect_true(est.optimal_decode_gpus >= 4 && est.optimal_decode_gpus <= 7,
                "optimal_decode_gpus: " + std::to_string(est.optimal_decode_gpus));
    expect_true(!est.reason.empty(), "reason should not be empty");
  }

  // Test 3: Low-bandwidth → should NOT disaggregate (KV transfer too slow)
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 10.0;
    input.profile.prefill_fraction = 0.6;
    input.profile.prefill_contention = 1.5;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 10.0;

    auto est = hotpath::estimate_disaggregation(input);
    expect_true(!est.should_disaggregate,
                "low-bandwidth should NOT disaggregate: " + est.reason);
    expect_true(est.min_bandwidth_gbps > 0.0,
                "min_bandwidth should be > 0 Gbps: " +
                std::to_string(est.min_bandwidth_gbps));
  }

  // Verify all estimates have populated fields
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2048;
    input.profile.median_output_tokens = 512;
    input.profile.request_rate = 20.0;
    input.profile.prefill_contention = 0.8;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;

    auto est = hotpath::estimate_disaggregation(input);
    expect_true(est.mono_throughput_rps > 0, "mono throughput should be > 0");
    expect_true(est.mono_p99_ttft_ms > 0, "mono p99 ttft should be > 0");
    expect_true(est.disagg_throughput_rps > 0, "disagg throughput should be > 0");
    expect_true(est.optimal_prefill_gpus + est.optimal_decode_gpus == 8,
                "GPU split should sum to 8");
  }

  std::cerr << "test_disagg_model: all tests passed\n";
  return 0;
}
