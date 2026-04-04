#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/report.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main() {
  hotpath::ServeReportData data;
  data.model_name = "meta-llama/Llama-3-70B";
  data.engine = "vllm";
  data.gpu_info = "8x NVIDIA A100";
  data.total_requests = 847;
  data.duration_seconds = 60.0;
  data.throughput_rps = 14.1;

  data.queue_p50 = 2.1; data.queue_p90 = 8.4; data.queue_p99 = 41.2;
  data.prefill_p50 = 12.3; data.prefill_p90 = 45.1; data.prefill_p99 = 112.8;
  data.decode_total_p50 = 184.2; data.decode_total_p90 = 412.5; data.decode_total_p99 = 891.3;
  data.decode_per_token_p50 = 4.2; data.decode_per_token_p90 = 5.1; data.decode_per_token_p99 = 8.9;
  data.e2e_p50 = 201.4; data.e2e_p90 = 458.3; data.e2e_p99 = 983.1;

  data.prefill_compute_pct = 31.2;
  data.decode_compute_pct = 48.1;
  data.other_idle_pct = 20.7;

  data.cache_hit_rate = 0.412;
  data.avg_cache_usage = 73.2;
  data.evictions = 12;

  data.unique_prefixes = 23;
  data.cacheable_tokens_pct = 72.3;

  data.should_disaggregate = true;
  data.optimal_p = 1;
  data.optimal_d = 3;
  data.projected_throughput_pct = 40.0;
  data.projected_throughput_rps = 19.8;
  data.mono_p99_ttft = 89.0;
  data.disagg_p99_ttft = 52.0;
  data.min_bandwidth_gbps = 50.0;

  const std::string report = hotpath::render_serve_report(data);

  // Verify key sections
  expect_true(contains(report, "Latency"), "missing Latency section");
  expect_true(contains(report, "GPU Phase"), "missing GPU Phase section");
  expect_true(contains(report, "KV Cache"), "missing KV Cache section");
  expect_true(contains(report, "Disaggregation"), "missing Disaggregation section");
  expect_true(contains(report, "Prefix Sharing"), "missing Prefix Sharing section");

  // Verify key values
  expect_true(contains(report, "847"), "missing request count");
  expect_true(contains(report, "14.1"), "missing throughput");
  expect_true(contains(report, "DISAGGREGATE"), "missing recommendation");
  expect_true(contains(report, "1:3"), "missing P:D ratio");
  expect_true(contains(report, "meta-llama"), "missing model name");

  // Test monolithic recommendation
  data.should_disaggregate = false;
  const std::string mono_report = hotpath::render_serve_report(data);
  expect_true(contains(mono_report, "MONOLITHIC"), "should show MONOLITHIC");

  std::cerr << "test_serve_report: all tests passed\n";
  return 0;
}
