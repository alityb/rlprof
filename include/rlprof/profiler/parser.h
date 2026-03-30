#pragma once

#include <filesystem>
#include <vector>

#include "rlprof/profiler/kernel_record.h"

namespace rlprof::profiler {

std::vector<KernelRecord> parse_nsys_sqlite(const std::filesystem::path& path);

}  // namespace rlprof::profiler
