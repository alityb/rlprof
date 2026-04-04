#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/recommender.h"

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
  const std::string model = "meta-llama/Llama-3-70B";

  // Test "should disaggregate" case
  {
    hotpath::DisaggEstimate est;
    est.should_disaggregate = true;
    est.optimal_prefill_gpus = 2;
    est.optimal_decode_gpus = 6;
    est.throughput_improvement = 1.4;
    est.mono_throughput_rps = 10.0;
    est.disagg_throughput_rps = 14.0;
    est.mono_p99_ttft_ms = 89.0;
    est.disagg_p99_ttft_ms = 52.0;
    est.mono_p99_itl_ms = 5.0;
    est.disagg_p99_itl_ms = 3.0;
    est.kv_transfer_overhead_ms = 10.0;
    est.min_bandwidth_gbps = 50.0;
    est.reason = "test reason";

    // vLLM config
    const auto vllm = hotpath::generate_vllm_config(est, model);
    expect_true(contains(vllm, "kv_producer"), "vllm config should have kv_producer");
    expect_true(contains(vllm, "kv_consumer"), "vllm config should have kv_consumer");
    expect_true(contains(vllm, model), "vllm config should contain model name");
    expect_true(contains(vllm, "#!/bin/bash"), "vllm config should be a shell script");

    // llm-d config
    const auto llmd = hotpath::generate_llmd_config(est, model);
    expect_true(contains(llmd, "prefill:"), "llmd config should have prefill section");
    expect_true(contains(llmd, "decode:"), "llmd config should have decode section");
    expect_true(contains(llmd, "replicas: 2"), "llmd should have 2 prefill replicas");
    expect_true(contains(llmd, "replicas: 6"), "llmd should have 6 decode replicas");

    // Dynamo config
    const auto dynamo = hotpath::generate_dynamo_config(est, model);
    expect_true(contains(dynamo, "disaggregated"), "dynamo should be disaggregated");
    expect_true(contains(dynamo, model), "dynamo should contain model");

    // Summary
    const auto summary = hotpath::generate_summary(est, model);
    expect_true(contains(summary, "DISAGGREGATE"), "summary should recommend disaggregation");
    expect_true(contains(summary, "2:6"), "summary should show P:D ratio");
  }

  // Test "should not disaggregate" case
  {
    hotpath::DisaggEstimate est;
    est.should_disaggregate = false;
    est.optimal_prefill_gpus = 1;
    est.optimal_decode_gpus = 7;
    est.mono_throughput_rps = 50.0;
    est.disagg_throughput_rps = 48.0;
    est.throughput_improvement = 0.96;
    est.reason = "Short prompts — prefill is not a bottleneck.";

    const auto vllm = hotpath::generate_vllm_config(est, model);
    expect_true(!contains(vllm, "kv_producer"), "monolithic should not have kv_producer");
    expect_true(contains(vllm, "Monolithic"), "should say monolithic");

    const auto llmd = hotpath::generate_llmd_config(est, model);
    expect_true(contains(llmd, "monolithic"), "llmd should say monolithic");

    const auto dynamo = hotpath::generate_dynamo_config(est, model);
    expect_true(contains(dynamo, "monolithic"), "dynamo should say monolithic");

    const auto summary = hotpath::generate_summary(est, model);
    expect_true(contains(summary, "MONOLITHIC"), "summary should recommend monolithic");
  }

  std::cerr << "test_recommender: all tests passed\n";
  return 0;
}
