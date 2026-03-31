#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace rlprof {

enum class ValidationStatus {
  kPass,
  kWarn,
  kFail,
};

struct ValidationCheck {
  std::string name;
  ValidationStatus status = ValidationStatus::kFail;
  std::string detail;
};

std::vector<ValidationCheck> validate_profile(const std::filesystem::path& db_path);
std::string render_validation_report(
    const std::filesystem::path& db_path,
    const std::vector<ValidationCheck>& checks,
    bool color);

}  // namespace rlprof
