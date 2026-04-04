#include "hotpath/traffic_replayer.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace hotpath {
namespace {

// Simple JSON string extraction (no dependency on JSON library)
std::string extract_json_string(const std::string& json, const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  auto pos = json.find(pattern);
  if (pos == std::string::npos) return "";

  // Skip past key and colon
  pos = json.find(':', pos + pattern.size());
  if (pos == std::string::npos) return "";
  pos++;

  // Skip whitespace
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

  if (pos >= json.size()) return "";

  if (json[pos] == '"') {
    // String value
    pos++;
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
      if (json[pos] == '\\' && pos + 1 < json.size()) {
        pos++;
        if (json[pos] == '"') result += '"';
        else if (json[pos] == 'n') result += '\n';
        else if (json[pos] == '\\') result += '\\';
        else result += json[pos];
      } else {
        result += json[pos];
      }
      pos++;
    }
    return result;
  }

  return "";
}

int extract_json_int(const std::string& json, const std::string& key, int default_val) {
  const std::string pattern = "\"" + key + "\"";
  auto pos = json.find(pattern);
  if (pos == std::string::npos) return default_val;

  pos = json.find(':', pos + pattern.size());
  if (pos == std::string::npos) return default_val;
  pos++;

  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

  if (pos >= json.size() || !std::isdigit(json[pos])) return default_val;

  std::string num;
  while (pos < json.size() && std::isdigit(json[pos])) {
    num += json[pos++];
  }
  return std::stoi(num);
}

// Extract first human message from ShareGPT conversations array
std::string extract_sharegpt_prompt(const std::string& json) {
  // Look for "from": "human" followed by "value": "..."
  auto pos = json.find("\"from\"");
  while (pos != std::string::npos) {
    auto from_val = extract_json_string(json.substr(pos), "from");
    if (from_val == "human") {
      auto value_pos = json.find("\"value\"", pos);
      if (value_pos != std::string::npos) {
        return extract_json_string(json.substr(value_pos), "value");
      }
    }
    pos = json.find("\"from\"", pos + 1);
  }
  return "";
}

}  // namespace

std::vector<ReplayRequest> load_jsonl(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open JSONL file: " + path.string());
  }

  std::vector<ReplayRequest> requests;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    ReplayRequest req;
    req.prompt = extract_json_string(line, "prompt");
    req.max_tokens = extract_json_int(line, "max_tokens", 256);

    if (!req.prompt.empty()) {
      requests.push_back(std::move(req));
    }
  }

  return requests;
}

std::vector<ReplayRequest> load_sharegpt(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open ShareGPT file: " + path.string());
  }

  std::vector<ReplayRequest> requests;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    ReplayRequest req;
    req.prompt = extract_sharegpt_prompt(line);
    req.max_tokens = 256;

    if (!req.prompt.empty()) {
      requests.push_back(std::move(req));
    }
  }

  return requests;
}

std::string build_request_body(const ReplayRequest& req, const std::string& model) {
  std::ostringstream ss;
  ss << "{";
  ss << "\"model\": \"" << model << "\", ";
  ss << "\"prompt\": \"";
  // Escape the prompt for JSON
  for (char c : req.prompt) {
    if (c == '"') ss << "\\\"";
    else if (c == '\\') ss << "\\\\";
    else if (c == '\n') ss << "\\n";
    else if (c == '\r') ss << "\\r";
    else if (c == '\t') ss << "\\t";
    else ss << c;
  }
  ss << "\", ";
  ss << "\"max_tokens\": " << req.max_tokens << ", ";
  ss << "\"stream\": true";
  ss << "}";
  return ss.str();
}

}  // namespace hotpath
