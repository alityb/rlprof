#pragma once

#include <string>

#include "hotpath/workload_classifier.h"

namespace hotpath {

struct DisaggModelInput {
    WorkloadProfile profile;
    int total_gpus = 8;
    double network_bandwidth_gbps = 100.0;
    double avg_kv_transfer_bytes = 0.0;
};

struct DisaggEstimate {
    double mono_throughput_rps = 0.0;
    double mono_p99_ttft_ms = 0.0;
    double mono_p99_itl_ms = 0.0;

    int optimal_prefill_gpus = 0;
    int optimal_decode_gpus = 0;
    double disagg_throughput_rps = 0.0;
    double disagg_p99_ttft_ms = 0.0;
    double disagg_p99_itl_ms = 0.0;
    double kv_transfer_overhead_ms = 0.0;

    bool should_disaggregate = false;
    std::string reason;
    double throughput_improvement = 1.0;
    double min_bandwidth_gbps = 0.0;
};

DisaggEstimate estimate_disaggregation(const DisaggModelInput& input);

}  // namespace hotpath
