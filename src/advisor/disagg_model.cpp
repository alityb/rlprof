#include "hotpath/disagg_model.h"

#include <algorithm>
#include <cmath>

namespace hotpath {
namespace {

// Simplified M/G/1 queue approximation
// Returns (throughput_rps, p99_wait_ms) for a queue with given params
struct QueueResult {
    double throughput_rps;
    double p99_wait_ms;
};

QueueResult mg1_estimate(double service_time_ms, int num_servers, double arrival_rate) {
  const double mu = static_cast<double>(num_servers) / service_time_ms * 1000.0;
  const double rho = arrival_rate / mu;

  if (rho >= 0.95) {
    // Near saturation
    return {mu * 0.95, service_time_ms * 20.0};
  }

  const double max_throughput = mu;
  // M/G/1 P99 approximation: service_time * (1 + rho/(1-rho)) * 2.33
  const double p99_wait = service_time_ms * (1.0 + rho / (1.0 - rho + 0.001)) * 2.33;

  return {max_throughput, p99_wait};
}

}  // namespace

DisaggEstimate estimate_disaggregation(const DisaggModelInput& input) {
  DisaggEstimate est;
  const auto& prof = input.profile;

  // Estimate service times from workload profile
  // Prefill time scales with prompt tokens
  const double prefill_time_ms = prof.median_prompt_tokens * 0.01;  // ~10us per token
  // Decode time scales with output tokens
  const double decode_step_ms = 5.0;  // ~5ms per decode step
  const double total_decode_ms = prof.median_output_tokens * decode_step_ms;

  // Monolithic baseline: service_time = prefill + decode
  const double mono_service_ms = prefill_time_ms + total_decode_ms;
  const auto mono = mg1_estimate(mono_service_ms, input.total_gpus, prof.request_rate);
  est.mono_throughput_rps = mono.throughput_rps;
  est.mono_p99_ttft_ms = prefill_time_ms * (1.0 + prof.prefill_contention);
  est.mono_p99_itl_ms = decode_step_ms * (1.0 + prof.prefill_contention * 0.5);

  // KV transfer overhead
  const double bytes_per_gbps = 1e9 / 8.0;  // bytes per second at 1 Gbps
  double kv_bytes = input.avg_kv_transfer_bytes;
  if (kv_bytes <= 0.0) {
    // Estimate: ~256 bytes per token per layer, 32 layers typical
    kv_bytes = prof.median_prompt_tokens * 256.0 * 32.0;
  }
  est.kv_transfer_overhead_ms =
      (kv_bytes / (input.network_bandwidth_gbps * bytes_per_gbps)) * 1000.0;

  // Minimum bandwidth: transfer must complete in < prefill_time to be worthwhile
  if (prefill_time_ms > 0) {
    est.min_bandwidth_gbps =
        (kv_bytes / (prefill_time_ms / 1000.0)) / bytes_per_gbps;
  }

  // Search for optimal P:D split
  double best_throughput = 0.0;
  int best_p = 1;
  int best_d = input.total_gpus - 1;

  for (int p = 1; p < input.total_gpus; ++p) {
    const int d = input.total_gpus - p;
    const auto prefill_q = mg1_estimate(prefill_time_ms, p, prof.request_rate);
    const auto decode_q = mg1_estimate(total_decode_ms, d, prof.request_rate);
    const double throughput = std::min(prefill_q.throughput_rps, decode_q.throughput_rps);

    if (throughput > best_throughput) {
      best_throughput = throughput;
      best_p = p;
      best_d = d;
    }
  }

  est.optimal_prefill_gpus = best_p;
  est.optimal_decode_gpus = best_d;
  est.disagg_throughput_rps = best_throughput;
  est.disagg_p99_ttft_ms = prefill_time_ms + est.kv_transfer_overhead_ms;
  est.disagg_p99_itl_ms = decode_step_ms;  // No prefill contention in disagg

  // Decision logic
  est.throughput_improvement = est.mono_throughput_rps > 0
      ? est.disagg_throughput_rps / est.mono_throughput_rps
      : 1.0;

  const bool kv_transfer_acceptable = est.kv_transfer_overhead_ms < prefill_time_ms * 0.5;
  const bool throughput_gain = est.throughput_improvement > 1.1;
  const bool significant_contention = prof.prefill_contention > 0.5;
  const bool needs_disagg = prof.median_prompt_tokens >= 1024;

  est.should_disaggregate =
      needs_disagg && kv_transfer_acceptable && (throughput_gain || significant_contention);

  if (est.should_disaggregate) {
    est.reason = "Workload has long prefill (" +
                 std::to_string(static_cast<int>(prof.median_prompt_tokens)) +
                 " tokens) with " +
                 std::to_string(static_cast<int>(est.throughput_improvement * 100 - 100)) +
                 "% throughput improvement. Optimal split: " +
                 std::to_string(best_p) + ":" + std::to_string(best_d) + " (P:D).";
  } else {
    if (!needs_disagg) {
      est.reason = "Short prompts (" +
                   std::to_string(static_cast<int>(prof.median_prompt_tokens)) +
                   " tokens) — prefill is not a bottleneck.";
    } else if (!kv_transfer_acceptable) {
      est.reason = "KV transfer overhead (" +
                   std::to_string(static_cast<int>(est.kv_transfer_overhead_ms)) +
                   "ms) too high. Need >= " +
                   std::to_string(static_cast<int>(std::ceil(est.min_bandwidth_gbps))) +
                   " Gbps bandwidth.";
    } else {
      est.reason = "Throughput improvement insufficient (" +
                   std::to_string(static_cast<int>(est.throughput_improvement * 100)) +
                   "%).";
    }
  }

  return est;
}

}  // namespace hotpath
