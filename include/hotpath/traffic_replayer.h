#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace hotpath {

struct ReplayRequest {
    std::string prompt;
    int max_tokens = 256;
    std::string model;  // optional: for /v1/completions
};

struct ReplayResult {
    std::string request_id;
    int64_t send_us = 0;
    int64_t first_token_us = 0;
    int64_t completion_us = 0;
    int tokens_generated = 0;
    bool success = false;
    std::string error;
};

struct ReplayConfig {
    std::string endpoint = "http://localhost:8000";
    int max_concurrency = 16;
    double rate_limit_rps = 0.0;  // 0 = unlimited
    std::string model;
};

std::vector<ReplayRequest> load_jsonl(const std::filesystem::path& path);
std::vector<ReplayRequest> load_sharegpt(const std::filesystem::path& path);

// Build the HTTP request body JSON for a given request
std::string build_request_body(const ReplayRequest& req, const std::string& model);

}  // namespace hotpath
