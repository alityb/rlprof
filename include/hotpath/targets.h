#pragma once

#include <string>
#include <vector>

#include "hotpath/remote.h"

namespace hotpath {

struct SavedTarget {
  std::string name;
  std::string host;
  std::string workdir = "~/hotpath";
  std::string python_executable;
  std::string vllm_executable;
};

std::vector<SavedTarget> list_targets();
void save_target(const SavedTarget& target);
bool remove_target(const std::string& name);
std::string render_targets(const std::vector<SavedTarget>& targets);
RemoteTarget resolve_target(const std::string& spec, const std::string& workdir_override = "");
std::string bootstrap_target_command(const RemoteTarget& target, const std::string& local_repo_root);

}  // namespace hotpath
