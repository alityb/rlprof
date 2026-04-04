#pragma once

#include <filesystem>
#include <string>

namespace hotpath {

struct ServeProfileOptions {
    std::string endpoint = "http://localhost:8000";
    int duration_seconds = 60;
    std::string traffic_path;
    std::string output = ".hotpath/serve_run";
    bool use_nsys = false;
    std::string engine = "vllm";  // vllm | sglang
};

int run_serve_profile(const ServeProfileOptions& opts);

}  // namespace hotpath
