#pragma once

#include <string>
#include <utility>
#include <vector>

#include "hotpath/batch_analyzer.h"

namespace hotpath {

struct SglangMetrics {
    double num_running_req = 0;
    double num_waiting_req = 0;
    double token_usage = 0;
    double cache_hit_rate = 0;
    double num_total_tokens = 0;
};

// Parse SGLang Prometheus-format metrics text into key-value pairs
std::vector<std::pair<std::string, double>> parse_sglang_metrics_text(const std::string& text);

// Parse into SglangMetrics struct
SglangMetrics parse_sglang_metrics(const std::string& text);

// Convert SGLang metrics to a MetricSnapshot for use by downstream analyzers
MetricSnapshot sglang_to_snapshot(const SglangMetrics& m, int64_t timestamp_us);

}  // namespace hotpath
