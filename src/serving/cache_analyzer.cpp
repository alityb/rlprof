#include "hotpath/cache_analyzer.h"

#include <algorithm>
#include <numeric>

namespace hotpath {

CacheAnalysis analyze_cache(const std::vector<RequestTrace>& traces,
                            const std::vector<MetricSnapshot>& snapshots) {
  CacheAnalysis result;

  // Cache hit rate from request traces
  int64_t total_cached = 0;
  int64_t total_prompt = 0;
  for (const auto& t : traces) {
    total_cached += t.cached_tokens;
    total_prompt += t.prompt_tokens;

    // Histogram bucket assignment
    double hit_frac = 0.0;
    if (t.prompt_tokens > 0) {
      hit_frac = static_cast<double>(t.cached_tokens) / t.prompt_tokens;
    }
    if (hit_frac <= 0.0) {
      result.hit_rate_histogram[0]++;
    } else if (hit_frac <= 0.25) {
      result.hit_rate_histogram[1]++;
    } else if (hit_frac <= 0.50) {
      result.hit_rate_histogram[2]++;
    } else if (hit_frac <= 0.75) {
      result.hit_rate_histogram[3]++;
    } else {
      result.hit_rate_histogram[4]++;
    }
  }
  if (total_prompt > 0) {
    result.cache_hit_rate = static_cast<double>(total_cached) / total_prompt;
  }

  // Cache usage from metric snapshots
  if (!snapshots.empty()) {
    double sum_usage = 0.0;
    double max_usage = 0.0;
    double max_preemption = 0.0;
    double min_preemption = snapshots[0].preemption_total;
    double pressure_samples = 0.0;

    for (const auto& s : snapshots) {
      sum_usage += s.cache_usage;
      max_usage = std::max(max_usage, s.cache_usage);
      max_preemption = std::max(max_preemption, s.preemption_total);
      min_preemption = std::min(min_preemption, s.preemption_total);
      if (s.cache_usage > 90.0) {
        pressure_samples += 1.0;
      }
    }

    result.avg_cache_usage = sum_usage / static_cast<double>(snapshots.size());
    result.peak_cache_usage = max_usage;
    // Each snapshot is ~1 second apart
    result.cache_pressure_seconds = pressure_samples;
    result.eviction_count = static_cast<int>(max_preemption - min_preemption);
  }

  return result;
}

}  // namespace hotpath
