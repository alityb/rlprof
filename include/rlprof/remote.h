#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace rlprof {

struct RemoteTarget {
  std::string host;
  std::string workdir = "~/rlprof";
  std::string python_executable;
  std::string vllm_executable;
};

bool has_remote_target(const RemoteTarget& target);
std::string remote_join(const RemoteTarget& target, const std::filesystem::path& local_path);
std::string remote_cli_command(
    const RemoteTarget& target,
    const std::vector<std::string>& args);
std::string remote_file_exists_command(
    const RemoteTarget& target,
    const std::string& remote_path);
std::string remote_checksum_command(
    const RemoteTarget& target,
    const std::string& remote_path);
std::string remote_copy_from_command(
    const RemoteTarget& target,
    const std::string& remote_path,
    const std::filesystem::path& local_path);
std::string remote_tail_command(
    const RemoteTarget& target,
    const std::string& remote_path,
    int lines = 120);
std::string remote_epoch_ms_command(const RemoteTarget& target);

}  // namespace rlprof
