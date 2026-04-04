#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "hotpath/request_trace.h"

namespace hotpath {

// Export request traces to OTLP JSON format
std::string export_otlp_json(const std::vector<RequestTrace>& traces,
                             const std::string& service_name = "hotpath");

void export_otlp_file(const std::vector<RequestTrace>& traces,
                      const std::filesystem::path& output_path,
                      const std::string& service_name = "hotpath");

}  // namespace hotpath
