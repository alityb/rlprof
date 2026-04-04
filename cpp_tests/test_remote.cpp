#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "hotpath/remote.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const hotpath::RemoteTarget target{
      .host = "ubuntu@a10g-box",
      .workdir = "/srv/hotpath",
      .python_executable = "/opt/venvs/hotpath/bin/python",
      .vllm_executable = "/opt/venvs/hotpath/bin/vllm",
  };

  expect_true(hotpath::has_remote_target(target), "expected remote target to be enabled");
  expect_true(
      hotpath::remote_join(target, std::filesystem::path(".hotpath/qwen3_8b_prod")) ==
          "/srv/hotpath/.hotpath/qwen3_8b_prod",
      "expected remote output path to map to remote workdir");

  const std::string profile_command =
      hotpath::remote_cli_command(
          target,
          {"profile", "--model", "Qwen/Qwen3-8B", "--output", "/srv/hotpath/.hotpath/qwen3_8b_prod"});
  expect_true(
      profile_command.find("ssh 'ubuntu@a10g-box'") != std::string::npos,
      "expected ssh host in remote command");
  expect_true(
      profile_command.find("./build/hotpath") != std::string::npos &&
          profile_command.find("/srv/hotpath/.hotpath/qwen3_8b_prod") != std::string::npos,
      "expected remote cli invocation");
  expect_true(
      profile_command.find("RLPROF_PYTHON_EXECUTABLE") != std::string::npos &&
          profile_command.find("RLPROF_VLLM_EXECUTABLE") != std::string::npos,
      "expected remote environment exports");

  const std::string attach_command =
      hotpath::remote_cli_command(
          target,
          {"profile",
           "--attach",
           "http://127.0.0.1:8070",
           "--attach-pid",
           "215839",
           "--prompts",
           "1",
           "--rollouts",
           "1",
           "--output",
           "/srv/hotpath/.hotpath/attach_remote"});
  expect_true(
      attach_command.find("profile") != std::string::npos &&
          attach_command.find("attach_remote") != std::string::npos &&
          attach_command.find("215839") != std::string::npos,
      "expected remote attach argument forwarding: " + attach_command);

  const std::string copy_command =
      hotpath::remote_copy_from_command(
          target,
          "/srv/hotpath/.hotpath/qwen3_8b_prod.db",
          ".hotpath/qwen3_8b_prod.db");
  expect_true(
      copy_command.find("scp -q") != std::string::npos,
      "expected scp copy command");
  expect_true(
      copy_command.find("ubuntu@a10g-box:/srv/hotpath/.hotpath/qwen3_8b_prod.db") != std::string::npos,
      "expected remote artifact path");

  const std::string epoch_command = hotpath::remote_epoch_ms_command(target);
  expect_true(
      epoch_command.find("date +%s%3N") != std::string::npos,
      "expected remote epoch command");

  return 0;
}
