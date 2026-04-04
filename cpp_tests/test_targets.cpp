#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "hotpath/targets.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

template <typename Fn>
void expect_throw(Fn&& fn, const std::string& message) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  std::cerr << message << "\n";
  std::exit(1);
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "hotpath_test_targets";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root / "config");
  setenv("XDG_CONFIG_HOME", (temp_root / "config").c_str(), 1);

  hotpath::save_target({
      .name = "a10g",
      .host = "ubuntu@a10g-box",
      .workdir = "/srv/hotpath",
      .python_executable = "/opt/venvs/hotpath/bin/python",
      .vllm_executable = "/opt/venvs/hotpath/bin/vllm",
  });

  const auto listed = hotpath::list_targets();
  expect_true(listed.size() == 1, "expected one saved target");
  expect_true(listed[0].name == "a10g", "expected saved target name");

  const auto resolved = hotpath::resolve_target("a10g");
  expect_true(resolved.host == "ubuntu@a10g-box", "expected resolved host");
  expect_true(resolved.workdir == "/srv/hotpath", "expected resolved workdir");
  expect_true(
      resolved.python_executable == "/opt/venvs/hotpath/bin/python",
      "expected resolved python executable");
  expect_true(
      resolved.vllm_executable == "/opt/venvs/hotpath/bin/vllm",
      "expected resolved vllm executable");

  const auto direct = hotpath::resolve_target("ubuntu@direct-box", "/tmp/hotpath");
  expect_true(direct.host == "ubuntu@direct-box", "expected direct host");
  expect_true(direct.workdir == "/tmp/hotpath", "expected direct workdir");

  expect_throw(
      []() { static_cast<void>(hotpath::resolve_target("gpu-a10g", "/opt/hotpath")); },
      "expected unknown alias target to throw");

  const auto rendered = hotpath::render_targets(listed);
  expect_true(rendered.find("TARGETS") != std::string::npos, "expected target table header");
  expect_true(rendered.find("ubuntu@a10g-box") != std::string::npos, "expected target host");
  expect_true(
      rendered.find("/opt/venvs/hotpath/bin/python") != std::string::npos,
      "expected rendered python executable");

  const auto bootstrap = hotpath::bootstrap_target_command(resolved, "/home/ubuntu/hotpath");
  expect_true(bootstrap.find("cmake -S . -B build") != std::string::npos,
              "expected bootstrap build command");
  expect_true(bootstrap.find("cmake --build build --target hotpath") != std::string::npos,
              "expected bootstrap hotpath target build");

  expect_true(hotpath::remove_target("a10g"), "expected target removal");
  expect_true(hotpath::list_targets().empty(), "expected empty target registry");

  unsetenv("XDG_CONFIG_HOME");
  fs::remove_all(temp_root);
  return 0;
}
