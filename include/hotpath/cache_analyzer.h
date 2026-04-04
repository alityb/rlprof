#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "hotpath/batch_analyzer.h"
#include "hotpath/request_trace.h"

namespace hotpath {

struct CacheAnalysis {
    double cache_hit_rate = 0.0;
    double avg_cache_usage = 0.0;
    double peak_cache_usage = 0.0;
    double cache_pressure_seconds = 0.0;
    int eviction_count = 0;
    // histogram: 0%, 1-25%, 25-50%, 50-75%, 75-100% cache hit per request
    std::array<int, 5> hit_rate_histogram = {};
};

CacheAnalysis analyze_cache(const std::vector<RequestTrace>& traces,
                            const std::vector<MetricSnapshot>& snapshots);

}  // namespace hotpath
