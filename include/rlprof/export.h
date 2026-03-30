#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace rlprof {

std::vector<std::filesystem::path> export_profile(
    const std::filesystem::path& path,
    const std::string& format);

}  // namespace rlprof
