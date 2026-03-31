#pragma once

#include <filesystem>
#include <string>

namespace rlprof {

std::string render_manifest_json(const std::filesystem::path& db_path);
std::filesystem::path write_manifest(
    const std::filesystem::path& db_path,
    const std::filesystem::path& output_path = {});

std::string cleanup_artifacts(
    const std::filesystem::path& dir,
    int keep_count,
    bool compress,
    bool apply_changes);

}  // namespace rlprof
