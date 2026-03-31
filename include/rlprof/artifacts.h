#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace rlprof {

struct ArtifactEntry {
  std::string kind;
  std::filesystem::path path;
  bool exists = false;
};

std::vector<ArtifactEntry> profile_artifacts(const std::filesystem::path& db_path);
std::vector<ArtifactEntry> trace_artifacts(const std::filesystem::path& db_path);

std::string render_artifacts(
    const std::filesystem::path& db_path,
    const std::vector<ArtifactEntry>& artifacts);
std::string render_trace_artifacts(
    const std::filesystem::path& db_path,
    const std::vector<ArtifactEntry>& artifacts,
    bool metrics_only);

}  // namespace rlprof
