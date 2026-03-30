#pragma once

#include <string_view>

namespace rlprof::profiler {

std::string_view categorize(std::string_view kernel_name);

}  // namespace rlprof::profiler
