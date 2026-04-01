#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "rlprof/profiler/server.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_test_server";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root);
  const fs::path old_cwd = fs::current_path();
  fs::current_path(temp_root);

  const rlprof::profiler::ManagedServerState seed{
      .name = "warm-qwen",
      .model = "Qwen/Qwen2.5-0.5B-Instruct",
      .session_name = "rlprof_managed_warm_qwen",
      .server_url = "http://127.0.0.1:8123",
      .output_prefix = ".rlprof/server_warm_qwen",
      .log_path = ".rlprof/server_warm_qwen_server.log",
      .pid = 99999,
      .port = 8123,
      .tp = 1,
      .max_model_len = 3072,
      .trust_remote_code = false,
  };

  fs::create_directories(".rlprof/servers");
  {
    std::ofstream stream(".rlprof/servers/warm-qwen.cfg");
    stream << "name=" << seed.name << "\n";
    stream << "model=" << seed.model << "\n";
    stream << "session_name=" << seed.session_name << "\n";
    stream << "server_url=" << seed.server_url << "\n";
    stream << "output_prefix=" << seed.output_prefix.string() << "\n";
    stream << "log_path=" << seed.log_path.string() << "\n";
    stream << "pid=" << seed.pid << "\n";
    stream << "port=" << seed.port << "\n";
    stream << "tp=" << seed.tp << "\n";
    stream << "max_model_len=" << seed.max_model_len << "\n";
    stream << "trust_remote_code=false\n";
  }
  fs::create_directories(".rlprof/servers/warm-qwen.lock");
  {
    std::ofstream stream(".rlprof/servers/warm-qwen.lock/pid");
    stream << "99999\n";
  }

  const auto loaded = rlprof::profiler::load_managed_server("warm-qwen");
  expect_true(loaded.name == seed.name, "expected server name");
  expect_true(loaded.model == seed.model, "expected server model");
  expect_true(loaded.session_name == seed.session_name, "expected server session");
  expect_true(loaded.server_url == seed.server_url, "expected server url");
  expect_true(loaded.max_model_len == seed.max_model_len, "expected max model len");

  const auto listed = rlprof::profiler::list_managed_servers();
  expect_true(listed.empty(), "expected stale managed server to be pruned");
  expect_true(
      !fs::exists(".rlprof/servers/warm-qwen.cfg"),
      "expected stale managed server state to be removed");
  expect_true(
      !fs::exists(".rlprof/servers/warm-qwen.lock"),
      "expected stale managed server lock to be removed");
  expect_true(
      !rlprof::profiler::find_managed_server("warm-qwen").has_value(),
      "expected no managed server after prune");

  const auto rendered = rlprof::profiler::render_managed_servers(listed);
  expect_true(rendered.find("MANAGED SERVERS") != std::string::npos, "expected header");
  expect_true(rendered.find("max-len") != std::string::npos, "expected max-len column");

  fs::current_path(old_cwd);
  fs::remove_all(temp_root);
  return 0;
}
