#include "hotpath/log_parser.h"

#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace hotpath {
namespace {

// Parse timestamp from vLLM log lines like:
// INFO 01-15 14:23:45.123456 ... or INFO 2024-01-15 14:23:45.123456 ...
// Returns microseconds since an arbitrary epoch, or 0 if not parseable.
int64_t parse_timestamp(const std::string& line) {
  // Match HH:MM:SS.ffffff
  static const std::regex ts_re(R"((\d{2}):(\d{2}):(\d{2})\.(\d{1,6}))");
  std::smatch match;
  if (!std::regex_search(line, match, ts_re)) {
    return 0;
  }
  const int64_t hours = std::stoll(match[1].str());
  const int64_t minutes = std::stoll(match[2].str());
  const int64_t seconds = std::stoll(match[3].str());
  std::string frac = match[4].str();
  while (frac.size() < 6) frac += '0';
  const int64_t micros = std::stoll(frac);
  return ((hours * 3600 + minutes * 60 + seconds) * 1000000) + micros;
}

// Extract request ID patterns like: cmpl-xxx, req-xxx, or quoted request IDs
std::string extract_request_id(const std::string& line) {
  // Match common vLLM request ID patterns
  static const std::regex id_re(R"((cmpl-[a-zA-Z0-9-]+|req-[a-zA-Z0-9-]+|[a-f0-9]{32}))");
  std::smatch match;
  if (std::regex_search(line, match, id_re)) {
    return match[1].str();
  }
  return "";
}

int extract_token_count(const std::string& line, const std::string& keyword) {
  // Look for patterns like "prompt_tokens=512" or "512 tokens" or "tokens: 512"
  std::regex re(keyword + R"([=:\s]+(\d+))");
  std::smatch match;
  if (std::regex_search(line, match, re)) {
    return std::stoi(match[1].str());
  }
  return 0;
}

}  // namespace

std::vector<RequestTrace> parse_vllm_log_lines(const std::vector<std::string>& lines) {
  std::map<std::string, RequestTrace> traces;

  for (const auto& line : lines) {
    const std::string request_id = extract_request_id(line);
    if (request_id.empty()) continue;

    const int64_t ts = parse_timestamp(line);
    auto& trace = traces[request_id];
    if (trace.request_id.empty()) {
      trace.request_id = request_id;
    }

    // Detect event type from log line content
    if (line.find("Added request") != std::string::npos ||
        line.find("add_request") != std::string::npos) {
      trace.arrival_us = ts;
      trace.queue_start_us = ts;
      trace.events.push_back(RequestEvent{
          .event_type = "queue",
          .timestamp_us = ts,
          .detail = "{}",
      });

      // Try to extract prompt token count
      int pt = extract_token_count(line, "prompt_token");
      if (pt == 0) pt = extract_token_count(line, "num_prompt_token");
      if (pt > 0) trace.prompt_tokens = pt;
    } else if (line.find("Running:") != std::string::npos ||
               line.find("scheduled") != std::string::npos ||
               line.find("Scheduling") != std::string::npos) {
      if (trace.prefill_start_us == 0) {
        trace.prefill_start_us = ts;
        trace.events.push_back(RequestEvent{
            .event_type = "schedule",
            .timestamp_us = ts,
            .detail = "{}",
        });
      }
    } else if (line.find("Prefill") != std::string::npos ||
               line.find("prefill") != std::string::npos) {
      if (trace.prefill_start_us == 0) {
        trace.prefill_start_us = ts;
      }
      if (trace.prefill_end_us == 0 ||
          line.find("done") != std::string::npos ||
          line.find("finished") != std::string::npos ||
          line.find("complete") != std::string::npos) {
        trace.prefill_end_us = ts;
      }
      trace.events.push_back(RequestEvent{
          .event_type = "prefill",
          .timestamp_us = ts,
          .detail = "{}",
      });

      int pt = extract_token_count(line, "token");
      if (pt > 0 && trace.prompt_tokens == 0) trace.prompt_tokens = pt;
    } else if (line.find("Finished request") != std::string::npos ||
               line.find("request_output") != std::string::npos ||
               line.find("completed") != std::string::npos) {
      trace.completion_us = ts;
      trace.last_token_us = ts;
      if (trace.first_token_us == 0 && trace.prefill_end_us > 0) {
        trace.first_token_us = trace.prefill_end_us;
      }
      trace.status = "ok";
      trace.events.push_back(RequestEvent{
          .event_type = "decode",
          .timestamp_us = ts,
          .detail = "{}",
      });

      int ot = extract_token_count(line, "output_token");
      if (ot == 0) ot = extract_token_count(line, "generated");
      if (ot > 0) trace.output_tokens = ot;
    } else if (line.find("preempt") != std::string::npos ||
               line.find("Preempt") != std::string::npos) {
      trace.status = "preempted";
      trace.events.push_back(RequestEvent{
          .event_type = "preempt",
          .timestamp_us = ts,
          .detail = "{}",
      });
    } else if (line.find("cache_hit") != std::string::npos ||
               line.find("prefix_cache") != std::string::npos) {
      trace.events.push_back(RequestEvent{
          .event_type = "cache_hit",
          .timestamp_us = ts,
          .detail = "{}",
      });
      int ct = extract_token_count(line, "cached");
      if (ct > 0) trace.cached_tokens = ct;
    }
  }

  // Set default status for traces without explicit completion
  std::vector<RequestTrace> result;
  result.reserve(traces.size());
  for (auto& [id, trace] : traces) {
    if (trace.status.empty()) {
      trace.status = "ok";
    }
    result.push_back(std::move(trace));
  }

  return result;
}

std::vector<RequestTrace> parse_vllm_log(const std::filesystem::path& log_path) {
  std::ifstream file(log_path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open log file: " + log_path.string());
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(std::move(line));
  }

  return parse_vllm_log_lines(lines);
}

}  // namespace hotpath
