#pragma once

#include <string>

#include "hotpath/batch_analyzer.h"
#include "hotpath/cache_analyzer.h"
#include "hotpath/phase_analyzer.h"
#include "hotpath/prefix_analyzer.h"

namespace hotpath {

enum class WorkloadClass {
    PREFILL_HEAVY,
    DECODE_HEAVY,
    BALANCED,
    CACHE_FRIENDLY,
    SHORT_CONTEXT,
};

struct WorkloadProfile {
    WorkloadClass primary_class;
    double prefill_fraction = 0.0;
    double median_prompt_tokens = 0.0;
    double median_output_tokens = 0.0;
    double request_rate = 0.0;
    double prefix_sharing_rate = 0.0;
    double cache_hit_rate = 0.0;
    double prefill_contention = 0.0;
};

struct WorkloadClassifierInput {
    PhaseBreakdown phase;
    BatchAnalysis batch;
    CacheAnalysis cache;
    PrefixAnalysis prefix;
    double median_prompt_tokens = 0.0;
    double median_output_tokens = 0.0;
    double request_rate = 0.0;
    double median_decode_latency_us = 0.0;
    double p99_decode_latency_us = 0.0;
};

WorkloadProfile classify_workload(const WorkloadClassifierInput& input);

}  // namespace hotpath
