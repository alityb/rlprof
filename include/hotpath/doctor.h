#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hotpath {

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

struct RuntimeToolInfo {
  std::string requested_path;
  std::string resolved_path;
  std::string version;
  std::string detail;
  bool found = false;
  bool ambiguous = false;
  bool explicit_config = false;
};

struct RuntimeEnvironmentInfo {
  RuntimeToolInfo python;
  RuntimeToolInfo vllm;
  RuntimeToolInfo nsys;
  std::string gpu_name;
  std::string driver_version;
  std::string cuda_visible_devices;
  std::string bench_helper_detail;
  std::string nsys_environment_detail;
  std::int64_t cuda_visible_device_count = 0;
  bool bench_helper_ok = false;
  bool nsys_environment_ok = false;
};

RuntimeEnvironmentInfo inspect_runtime_environment();
std::vector<DoctorCheck> doctor_checks_from_environment(const RuntimeEnvironmentInfo& environment);
std::map<std::string, std::string> runtime_environment_metadata(
    const RuntimeEnvironmentInfo& environment);
std::vector<DoctorCheck> run_doctor();
std::string render_doctor_report(const std::vector<DoctorCheck>& checks, bool color);

}  // namespace hotpath
