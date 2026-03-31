#include <cstdlib>
#include <iostream>
#include <string>

#include "rlprof/doctor.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const auto rendered = rlprof::render_doctor_report(
      {
          rlprof::DoctorCheck{
              .name = "nsys",
              .status = rlprof::DoctorStatus::kPass,
              .detail = "version ok",
          },
          rlprof::DoctorCheck{
              .name = "clock policy",
              .status = rlprof::DoctorStatus::kWarn,
              .detail = "unlocked",
          },
          rlprof::DoctorCheck{
              .name = "vllm",
              .status = rlprof::DoctorStatus::kFail,
              .detail = "missing",
          },
      },
      false);
  expect_true(rendered.find("DOCTOR") != std::string::npos, "expected doctor header");
  expect_true(rendered.find("PASS") != std::string::npos, "expected pass row");
  expect_true(rendered.find("WARN") != std::string::npos, "expected warn row");
  expect_true(rendered.find("FAIL") != std::string::npos, "expected fail row");
  return 0;
}
