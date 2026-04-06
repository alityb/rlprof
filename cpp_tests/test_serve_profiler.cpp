#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "hotpath/serve_profiler.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << text;
}

}  // namespace

int main() {
  namespace fs = std::filesystem;

  expect_true(std::abs(hotpath::cache_usage_metric_to_percent("vllm:gpu_cache_usage_perc", 0.723) - 72.3) < 1e-9,
              "expected legacy vLLM cache usage to normalize to percent");
  expect_true(std::abs(hotpath::cache_usage_metric_to_percent("vllm:kv_cache_usage_perc", 0.341) - 34.1) < 1e-9,
              "expected vLLM 0.19 cache usage to normalize to percent");
  expect_true(std::abs(hotpath::cache_usage_metric_to_percent("other", 4.0) - 4.0) < 1e-9,
              "non-cache metrics should remain unchanged");

  const fs::path previous_cwd = fs::current_path();
  const fs::path temp_root = fs::temp_directory_path() / "hotpath_test_serve_profiler";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root);
  fs::current_path(temp_root);

  hotpath::ServeProfileOptions opts;
  opts.endpoint = "http://localhost:18000";
  opts.engine = "vllm";
  opts.output = ".hotpath/serve_run";

  const fs::path older = ".hotpath/video-server/vllm.log";
  const fs::path newer = ".hotpath/video-server/vllm.stderr.log";
  write_text(older, "old log\n");
  write_text(newer, "new log\n");
  fs::last_write_time(older, fs::file_time_type::clock::now() - std::chrono::seconds(5));
  fs::last_write_time(newer, fs::file_time_type::clock::now());

  const auto discovered = hotpath::discover_server_log_path(opts);
  expect_true(discovered.has_value(), "expected local vLLM log autodiscovery");
  expect_true(
      discovered->lexically_normal() == newer.lexically_normal(),
      "expected newest candidate log to be selected");

  setenv("HOTPATH_TEST_LISTENER_PID", "12345", 1);
  const fs::path matching_log = ".hotpath/video-server/vllm.stdout.log";
  const fs::path stderr_log = ".hotpath/video-server/vllm.stderr.log";
  write_text(matching_log, "stdout log\n");
  write_text(stderr_log, "stderr log\n");
  fs::last_write_time(matching_log, fs::file_time_type::clock::now() - std::chrono::seconds(2));
  fs::last_write_time(stderr_log, fs::file_time_type::clock::now() + std::chrono::seconds(2));
  setenv("HOTPATH_TEST_STDOUT_LOG_PATH", matching_log.c_str(), 1);
  setenv("HOTPATH_TEST_STDERR_LOG_PATH", stderr_log.c_str(), 1);

  const auto pid_filtered = hotpath::discover_server_log_path(opts);
  expect_true(pid_filtered.has_value(), "expected log autodiscovery with listener-backed stdout");
  expect_true(
      pid_filtered->lexically_normal() == matching_log.lexically_normal(),
      "expected autodiscovery to prefer the live listener stdout log");

  unsetenv("HOTPATH_TEST_STDOUT_LOG_PATH");
  unsetenv("HOTPATH_TEST_STDERR_LOG_PATH");
  const auto none_matching = hotpath::discover_server_log_path(opts);
  expect_true(
      !none_matching.has_value(),
      "expected no autodiscovery when the live listener is not writing logs to a regular file");
  unsetenv("HOTPATH_TEST_LISTENER_PID");

  fs::remove_all(".hotpath");

  hotpath::ServeProfileOptions custom_opts = opts;
  custom_opts.output = "artifacts/run";
  const fs::path custom_log = "artifacts/video-server/vllm.stderr.log";
  write_text(custom_log, "custom log\n");
  fs::last_write_time(custom_log, fs::file_time_type::clock::now() + std::chrono::seconds(10));
  const auto custom_discovered = hotpath::discover_server_log_path(custom_opts);
  expect_true(custom_discovered.has_value(), "expected custom output parent log autodiscovery");
  expect_true(
      custom_discovered->lexically_normal() == custom_log.lexically_normal(),
      "expected output-adjacent video log to win when newer");

  hotpath::ServeProfileOptions remote_opts = opts;
  remote_opts.endpoint = "http://10.0.0.12:8000";
  expect_true(
      !hotpath::discover_server_log_path(remote_opts).has_value(),
      "expected no autodiscovery for non-local endpoints");

  fs::current_path(previous_cwd);
  fs::remove_all(temp_root);
  std::cerr << "test_serve_profiler: all tests passed\n";
  return 0;
}
