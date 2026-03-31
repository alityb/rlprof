#pragma once

#include <string>
#include <vector>

namespace rlprof {

enum class DoctorStatus {
  kPass,
  kWarn,
  kFail,
};

struct DoctorCheck {
  std::string name;
  DoctorStatus status = DoctorStatus::kFail;
  std::string detail;
};

std::vector<DoctorCheck> run_doctor();
std::string render_doctor_report(const std::vector<DoctorCheck>& checks, bool color);

}  // namespace rlprof
