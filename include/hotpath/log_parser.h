#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "hotpath/request_trace.h"

namespace hotpath {

std::vector<RequestTrace> parse_vllm_log(const std::filesystem::path& log_path);
std::vector<RequestTrace> parse_vllm_log_lines(const std::vector<std::string>& lines);

}  // namespace hotpath
