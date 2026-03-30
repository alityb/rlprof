#include "rlprof/export.h"

#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rlprof/store.h"

namespace rlprof {
namespace {

std::string json_escape(const std::string& input) {
  std::string output;
  for (char ch : input) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      default:
        output.push_back(ch);
        break;
    }
  }
  return output;
}

std::string csv_escape(const std::string& input) {
  bool needs_quotes = false;
  for (char ch : input) {
    if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) {
    return input;
  }

  std::string output = "\"";
  for (char ch : input) {
    if (ch == '"') {
      output += "\"\"";
    } else {
      output.push_back(ch);
    }
  }
  output += "\"";
  return output;
}

std::string optional_json(const std::optional<double>& value) {
  return value.has_value() ? std::to_string(*value) : "null";
}

std::string optional_csv(const std::optional<double>& value) {
  return value.has_value() ? std::to_string(*value) : "";
}

void write_csv(const std::filesystem::path& path, const std::vector<std::string>& lines) {
  std::ofstream out(path);
  for (const std::string& line : lines) {
    out << line << "\n";
  }
}

}  // namespace

std::vector<std::filesystem::path> export_profile(
    const std::filesystem::path& path,
    const std::string& format) {
  const ProfileData profile = load_profile(path);
  std::vector<std::filesystem::path> outputs;

  if (format == "json") {
    const std::filesystem::path output_path = path.parent_path() / (path.stem().string() + ".json");
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"meta\": {\n";
    for (auto it = profile.meta.begin(); it != profile.meta.end(); ++it) {
      out << "    \"" << json_escape(it->first) << "\": \"" << json_escape(it->second) << "\"";
      out << (std::next(it) == profile.meta.end() ? "\n" : ",\n");
    }
    out << "  },\n";
    out << "  \"kernels\": [\n";
    for (std::size_t i = 0; i < profile.kernels.size(); ++i) {
      const auto& kernel = profile.kernels[i];
      out << "    {\"name\": \"" << json_escape(kernel.name)
          << "\", \"category\": \"" << json_escape(kernel.category)
          << "\", \"total_ns\": " << kernel.total_ns
          << ", \"calls\": " << kernel.calls
          << ", \"avg_ns\": " << kernel.avg_ns
          << ", \"min_ns\": " << kernel.min_ns
          << ", \"max_ns\": " << kernel.max_ns
          << ", \"registers\": " << kernel.registers
          << ", \"shared_mem\": " << kernel.shared_mem << "}";
      out << (i + 1 == profile.kernels.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"vllm_metrics\": [\n";
    for (std::size_t i = 0; i < profile.metrics.size(); ++i) {
      const auto& metric = profile.metrics[i];
      out << "    {\"sample_time\": " << metric.sample_time
          << ", \"metric\": \"" << json_escape(metric.metric)
          << "\", \"value\": " << metric.value << "}";
      out << (i + 1 == profile.metrics.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"vllm_metrics_summary\": [\n";
    for (std::size_t i = 0; i < profile.metrics_summary.size(); ++i) {
      const auto& summary = profile.metrics_summary[i];
      out << "    {\"metric\": \"" << json_escape(summary.metric)
          << "\", \"avg\": " << optional_json(summary.avg)
          << ", \"peak\": " << optional_json(summary.peak)
          << ", \"min\": " << optional_json(summary.min) << "}";
      out << (i + 1 == profile.metrics_summary.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"traffic_stats\": {\n";
    out << "    \"total_requests\": " << profile.traffic_stats.total_requests << ",\n";
    out << "    \"completion_length_mean\": " << optional_json(profile.traffic_stats.completion_length_mean) << ",\n";
    out << "    \"completion_length_p50\": " << optional_json(profile.traffic_stats.completion_length_p50) << ",\n";
    out << "    \"completion_length_p99\": " << optional_json(profile.traffic_stats.completion_length_p99) << ",\n";
    out << "    \"max_median_ratio\": " << optional_json(profile.traffic_stats.max_median_ratio) << ",\n";
    out << "    \"errors\": " << profile.traffic_stats.errors << "\n";
    out << "  }\n";
    out << "}\n";
    outputs.push_back(output_path);
    return outputs;
  }

  if (format == "csv") {
    const std::filesystem::path meta_path = path.parent_path() / (path.stem().string() + "_meta.csv");
    std::vector<std::string> meta_lines = {"key,value"};
    for (const auto& [key, value] : profile.meta) {
      meta_lines.push_back(csv_escape(key) + "," + csv_escape(value));
    }
    write_csv(meta_path, meta_lines);
    outputs.push_back(meta_path);

    const std::filesystem::path kernels_path = path.parent_path() / (path.stem().string() + "_kernels.csv");
    std::vector<std::string> kernel_lines = {
        "name,category,total_ns,calls,avg_ns,min_ns,max_ns,registers,shared_mem"};
    for (const auto& kernel : profile.kernels) {
      std::ostringstream line;
      line << csv_escape(kernel.name) << "," << csv_escape(kernel.category) << "," << kernel.total_ns << "," << kernel.calls
           << "," << kernel.avg_ns << "," << kernel.min_ns << "," << kernel.max_ns << ","
           << kernel.registers << "," << kernel.shared_mem;
      kernel_lines.push_back(line.str());
    }
    write_csv(kernels_path, kernel_lines);
    outputs.push_back(kernels_path);

    const std::filesystem::path metrics_path =
        path.parent_path() / (path.stem().string() + "_vllm_metrics.csv");
    std::vector<std::string> metrics_lines = {"sample_time,metric,value"};
    for (const auto& metric : profile.metrics) {
      std::ostringstream line;
      line << metric.sample_time << "," << csv_escape(metric.metric) << "," << metric.value;
      metrics_lines.push_back(line.str());
    }
    write_csv(metrics_path, metrics_lines);
    outputs.push_back(metrics_path);

    const std::filesystem::path summary_path =
        path.parent_path() / (path.stem().string() + "_vllm_metrics_summary.csv");
    std::vector<std::string> summary_lines = {"metric,avg,peak,min"};
    for (const auto& summary : profile.metrics_summary) {
      std::ostringstream line;
      line << csv_escape(summary.metric) << "," << optional_csv(summary.avg) << ","
           << optional_csv(summary.peak) << "," << optional_csv(summary.min);
      summary_lines.push_back(line.str());
    }
    write_csv(summary_path, summary_lines);
    outputs.push_back(summary_path);

    const std::filesystem::path traffic_path =
        path.parent_path() / (path.stem().string() + "_traffic_stats.csv");
    std::vector<std::string> traffic_lines = {
        "key,value",
        "total_requests," + std::to_string(profile.traffic_stats.total_requests),
        "completion_length_mean," + optional_csv(profile.traffic_stats.completion_length_mean),
        "completion_length_p50," + optional_csv(profile.traffic_stats.completion_length_p50),
        "completion_length_p99," + optional_csv(profile.traffic_stats.completion_length_p99),
        "max_median_ratio," + optional_csv(profile.traffic_stats.max_median_ratio),
        "errors," + std::to_string(profile.traffic_stats.errors),
    };
    write_csv(traffic_path, traffic_lines);
    outputs.push_back(traffic_path);
    return outputs;
  }

  throw std::runtime_error("unsupported export format: " + format);
}

}  // namespace rlprof
