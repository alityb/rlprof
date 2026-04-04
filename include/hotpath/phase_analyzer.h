#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hotpath/profiler/categorizer.h"

namespace hotpath {

struct KernelEntry {
    std::string name;
    profiler::KernelPhase phase;
    int64_t start_us;
    int64_t duration_us;
};

struct PhaseBreakdown {
    int64_t prefill_us = 0;
    int64_t decode_us = 0;
    int64_t unknown_us = 0;
    int64_t total_us = 0;
    double prefill_fraction = 0.0;
    double decode_fraction = 0.0;
    int prefill_kernel_count = 0;
    int decode_kernel_count = 0;
};

struct PhaseTimePoint {
    int64_t window_start_us;
    double prefill_fraction;
    double decode_fraction;
};

struct PhaseAnalysisResult {
    PhaseBreakdown breakdown;
    std::vector<PhaseTimePoint> time_series;
};

PhaseAnalysisResult analyze_phases(const std::vector<KernelEntry>& kernels);

}  // namespace hotpath
