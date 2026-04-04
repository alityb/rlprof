#pragma once

#include <filesystem>
#include <vector>

#include "hotpath/store.h"

namespace hotpath {

ProfileData aggregate_profiles(const std::vector<std::filesystem::path>& paths);

}  // namespace hotpath
