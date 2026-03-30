#include "rlprof/profiler/runner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <sqlite3.h>
#include <sys/types.h>

#include "rlprof/profiler/parser.h"
#include "rlprof/profiler/vllm_metrics.h"
#include "rlprof/store.h"

namespace rlprof::profiler {
namespace {

std::string run_command(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("failed to run command: " + command);
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  if (rc != 0) {
    throw std::runtime_error("command failed: " + command);
  }
  return output;
}

pid_t start_background_command(const std::string& command) {
  const std::string wrapped = "bash -lc '" + command + " > /tmp/rlprof_server.log 2>&1 & echo $!'";
  const std::string output = run_command(wrapped);
  return static_cast<pid_t>(std::stol(output));
}

bool sqlite_kernel_table_ready(const std::filesystem::path& path) {
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    if (db != nullptr) {
      sqlite3_close(db);
    }
    return false;
  }

  sqlite3_stmt* stmt = nullptr;
  constexpr const char* kSql =
      "SELECT 1 FROM sqlite_master "
      "WHERE type = 'table' AND name = 'CUPTI_ACTIVITY_KIND_KERNEL' LIMIT 1";
  const int prepare_rc = sqlite3_prepare_v2(db, kSql, -1, &stmt, nullptr);
  if (prepare_rc != SQLITE_OK) {
    sqlite3_close(db);
    return false;
  }

  const int step_rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return step_rc == SQLITE_ROW;
}

bool wait_for_kernel_table(const std::filesystem::path& path, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::exists(path) && sqlite_kernel_table_ready(path)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return std::filesystem::exists(path) && sqlite_kernel_table_ready(path);
}

void stop_process(pid_t pid) {
  if (pid > 0) {
    kill(pid, SIGTERM);
  }
}

bool server_ready(const std::string& server_url) {
  const std::string command = "curl -fsS " + server_url + "/metrics > /dev/null";
  return std::system(command.c_str()) == 0;
}

std::string trim(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::filesystem::path default_output_prefix(const ProfileConfig& config) {
  if (!config.output.empty()) {
    return config.output;
  }
  const auto now = std::chrono::system_clock::now().time_since_epoch().count();
  std::string model = config.model;
  std::replace(model.begin(), model.end(), '/', '_');
  return std::filesystem::path(".rlprof") / (model + "_" + std::to_string(now));
}

std::int64_t inferred_max_model_len(const ProfileConfig& config) {
  return std::max<std::int64_t>(2048, config.input_len + config.max_tokens);
}

std::string make_session_name(const std::filesystem::path& output_prefix) {
  std::string session = "rlprof_";
  const std::string stem = output_prefix.filename().string();
  for (unsigned char ch : stem) {
    session.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
  }
  return session;
}

}  // namespace

ProfileRunResult run_profile(const ProfileConfig& config) {
  const std::filesystem::path output_prefix = default_output_prefix(config);
  if (!output_prefix.parent_path().empty()) {
    std::filesystem::create_directories(output_prefix.parent_path());
  }

  const std::string server_url = "http://127.0.0.1:" + std::to_string(config.port);
  const std::string gpu_name = trim(run_command("nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1"));
  const std::string vllm_version = trim(run_command("vllm --version"));
  const std::int64_t max_model_len = inferred_max_model_len(config);
  const std::string session_name = make_session_name(output_prefix);

  const std::string serve_command =
      "VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile --trace=cuda,osrt "
      "--sample=none --cpuctxsw=none --export=sqlite "
      "--force-overwrite=true --session-new " + session_name +
      " --start-later=true -o " + output_prefix.string() +
      " vllm serve " + config.model +
      " --port " + std::to_string(config.port) +
      " --tensor-parallel-size " + std::to_string(config.tp) +
      " --max-model-len " + std::to_string(max_model_len) +
      (config.trust_remote_code ? " --trust-remote-code" : "");
  const pid_t server_pid = start_background_command(serve_command);

  std::atomic<bool> stop_flag{false};
  std::vector<MetricSample> metrics;
  std::thread metrics_thread;

  try {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(config.startup_timeout_s);
    bool ready = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (server_ready(server_url)) {
        ready = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!ready) {
      stop_process(server_pid);
      throw std::runtime_error(
          "vLLM server did not become ready within " +
          std::to_string(config.startup_timeout_s) + " seconds");
    }

    run_command("nsys start --session=" + session_name);

    metrics_thread = std::thread([&]() {
      metrics = poll_metrics(
          server_url,
          std::chrono::milliseconds(config.metrics_interval_ms),
          stop_flag);
    });

    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe");
    const std::string traffic_command =
        self.string() +
        " traffic --server " + server_url +
        " --prompts " + std::to_string(config.prompts) +
        " --rollouts-per-prompt " + std::to_string(config.rollouts) +
        " --max-tokens " + std::to_string(config.max_tokens) +
        " --min-tokens " + std::to_string(config.min_tokens) +
        " --input-len " + std::to_string(config.input_len);
    const std::string traffic_output = trim(run_command(traffic_command));

    stop_flag.store(true);
    if (metrics_thread.joinable()) {
      metrics_thread.join();
    }
    run_command("nsys stop --session=" + session_name);
    stop_process(server_pid);

    TrafficStats traffic_stats{
        .total_requests = 0,
        .completion_length_mean = std::nullopt,
        .completion_length_p50 = std::nullopt,
        .completion_length_p99 = std::nullopt,
        .max_median_ratio = std::nullopt,
        .errors = 0,
    };
    {
      const auto extract = [&](const std::string& key) -> std::optional<double> {
        const std::string needle = "\"" + key + "\":";
        const std::size_t start = traffic_output.find(needle);
        if (start == std::string::npos) {
          return std::nullopt;
        }
        std::size_t index = start + needle.size();
        while (index < traffic_output.size() && std::isspace(static_cast<unsigned char>(traffic_output[index]))) {
          ++index;
        }
        std::size_t end = index;
        while (end < traffic_output.size() &&
               (std::isdigit(static_cast<unsigned char>(traffic_output[end])) ||
                traffic_output[end] == '.' || traffic_output[end] == '-')) {
          ++end;
        }
        return std::stod(traffic_output.substr(index, end - index));
      };
      traffic_stats.total_requests = static_cast<std::int64_t>(extract("total_requests").value_or(0.0));
      traffic_stats.completion_length_mean = extract("completion_length_mean");
      traffic_stats.completion_length_p50 = extract("completion_length_p50");
      traffic_stats.completion_length_p99 = extract("completion_length_p99");
      traffic_stats.max_median_ratio = extract("max_median_ratio");
      traffic_stats.errors = static_cast<std::int64_t>(extract("errors").value_or(0.0));
    }

    std::filesystem::path sqlite_path = output_prefix;
    sqlite_path.replace_extension(".sqlite");
    if (!wait_for_kernel_table(sqlite_path, std::chrono::seconds(30))) {
      throw std::runtime_error("timed out waiting for nsys sqlite export readiness: " + sqlite_path.string());
    }
    const std::vector<KernelRecord> kernels = parse_nsys_sqlite(sqlite_path);
    const std::vector<MetricSummary> summaries = summarize_samples(metrics);

    ProfileData profile;
    profile.meta = {
        {"model_name", config.model},
        {"gpu_name", gpu_name},
        {"vllm_version", vllm_version},
        {"prompts", std::to_string(config.prompts)},
        {"rollouts", std::to_string(config.rollouts)},
        {"max_tokens", std::to_string(config.max_tokens)},
        {"min_tokens", std::to_string(config.min_tokens)},
        {"input_len", std::to_string(config.input_len)},
        {"max_model_len", std::to_string(max_model_len)},
        {"trust_remote_code", config.trust_remote_code ? "true" : "false"},
        {"port", std::to_string(config.port)},
        {"tp", std::to_string(config.tp)},
    };
    profile.kernels = kernels;
    profile.metrics = metrics;
    profile.metrics_summary = summaries;
    profile.traffic_stats = traffic_stats;

    std::filesystem::path db_path = output_prefix;
    db_path.replace_extension(".db");
    save_profile(db_path, profile);

    return ProfileRunResult{
        .db_path = db_path,
        .nsys_sqlite_path = sqlite_path,
        .meta = profile.meta,
        .kernels = kernels,
        .metrics = metrics,
        .traffic_stats = traffic_stats,
    };
  } catch (...) {
    stop_flag.store(true);
    if (metrics_thread.joinable()) {
      metrics_thread.join();
    }
    std::system(("nsys stop --session=" + session_name + " > /dev/null 2>&1").c_str());
    stop_process(server_pid);
    throw;
  }
}

}  // namespace rlprof::profiler
