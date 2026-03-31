#pragma once

#include <filesystem>
#include <vector>

#include "rlprof/store.h"

namespace rlprof {

ProfileData aggregate_profiles(const std::vector<std::filesystem::path>& paths);

}  // namespace rlprof
