#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hotpath {

struct MetricSnapshot {
    int64_t timestamp_us;
    double batch_size;
    double queue_depth;
    double preemption_total;
    double cache_usage;
};

struct BatchAnalysis {
    double avg_batch_size = 0;
    double p50_batch_size = 0;
    double p99_batch_size = 0;
    double avg_queue_depth = 0;
    double p99_queue_depth = 0;
    int total_preemptions = 0;
    double avg_cache_usage = 0;
    double peak_cache_usage = 0;
    std::vector<std::pair<int64_t, double>> batch_size_series;
    std::vector<std::pair<int64_t, double>> queue_depth_series;
};

BatchAnalysis analyze_batches(const std::vector<MetricSnapshot>& snapshots);

}  // namespace hotpath
