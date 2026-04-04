#pragma once

#include <filesystem>
#include <vector>

#include "hotpath/profiler/kernel_record.h"

namespace hotpath::profiler {

std::vector<KernelRecord> parse_nsys_sqlite(const std::filesystem::path& path);

}  // namespace hotpath::profiler
