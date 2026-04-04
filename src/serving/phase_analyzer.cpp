#include "hotpath/phase_analyzer.h"

#include <algorithm>
#include <map>

namespace hotpath {

PhaseAnalysisResult analyze_phases(const std::vector<KernelEntry>& kernels) {
  PhaseAnalysisResult result;
  auto& bd = result.breakdown;

  if (kernels.empty()) return result;

  int64_t min_start = kernels[0].start_us;
  int64_t max_end = kernels[0].start_us + kernels[0].duration_us;

  for (const auto& k : kernels) {
    switch (k.phase) {
      case profiler::KernelPhase::PREFILL:
        bd.prefill_us += k.duration_us;
        bd.prefill_kernel_count++;
        break;
      case profiler::KernelPhase::DECODE:
        bd.decode_us += k.duration_us;
        bd.decode_kernel_count++;
        break;
      case profiler::KernelPhase::UNKNOWN:
        bd.unknown_us += k.duration_us;
        break;
    }
    bd.total_us += k.duration_us;
    min_start = std::min(min_start, k.start_us);
    max_end = std::max(max_end, k.start_us + k.duration_us);
  }

  if (bd.total_us > 0) {
    bd.prefill_fraction = static_cast<double>(bd.prefill_us) / bd.total_us;
    bd.decode_fraction = static_cast<double>(bd.decode_us) / bd.total_us;
  }

  // Per-second time series
  const int64_t window_us = 1000000;  // 1 second
  for (int64_t ws = min_start; ws < max_end; ws += window_us) {
    const int64_t we = ws + window_us;
    int64_t prefill_in_window = 0;
    int64_t decode_in_window = 0;
    int64_t total_in_window = 0;

    for (const auto& k : kernels) {
      const int64_t ks = k.start_us;
      const int64_t ke = k.start_us + k.duration_us;
      const int64_t overlap_start = std::max(ks, ws);
      const int64_t overlap_end = std::min(ke, we);
      if (overlap_start >= overlap_end) continue;
      const int64_t overlap = overlap_end - overlap_start;
      total_in_window += overlap;
      if (k.phase == profiler::KernelPhase::PREFILL) prefill_in_window += overlap;
      else if (k.phase == profiler::KernelPhase::DECODE) decode_in_window += overlap;
    }

    PhaseTimePoint tp;
    tp.window_start_us = ws;
    tp.prefill_fraction = total_in_window > 0
        ? static_cast<double>(prefill_in_window) / total_in_window
        : 0.0;
    tp.decode_fraction = total_in_window > 0
        ? static_cast<double>(decode_in_window) / total_in_window
        : 0.0;
    result.time_series.push_back(tp);
  }

  return result;
}

}  // namespace hotpath
