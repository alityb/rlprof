#include <cstdlib>
#include <filesystem>
#include <fstream>
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

void write_script(
    const std::filesystem::path& path,
    const std::string& contents) {
  std::ofstream out(path);
  out << contents;
  out.close();
  std::filesystem::permissions(
      path,
      std::filesystem::perms::owner_exec | std::filesystem::perms::owner_read |
          std::filesystem::perms::owner_write,
      std::filesystem::perm_options::add);
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

  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_test_doctor";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root / "bin");

  write_script(
      temp_root / "bin" / "python3",
      "#!/bin/sh\n"
      "echo 'Python 3.12.9'\n");
  write_script(
      temp_root / "bin" / "python",
      "#!/bin/sh\n"
      "echo 'Python 3.11.7'\n");
  write_script(
      temp_root / "bin" / "vllm",
      "#!/bin/sh\n"
      "echo 'vLLM 0.18.0'\n");
  write_script(
      temp_root / "bin" / "nsys",
      "#!/bin/sh\n"
      "if [ \"$1\" = \"--version\" ]; then\n"
      "  echo 'NVIDIA Nsight Systems version 2025.3.2'\n"
      "elif [ \"$1\" = \"status\" ] && [ \"$2\" = \"-e\" ]; then\n"
      "  echo 'Timestamp counter supported: Yes'\n"
      "else\n"
      "  exit 1\n"
      "fi\n");
  write_script(
      temp_root / "bin" / "nvidia-smi",
      "#!/bin/sh\n"
      "case \"$*\" in\n"
      "  *name,driver_version*) echo 'NVIDIA A10G, 580.126.16' ;;\n"
      "  *index*) echo '0' ;;\n"
      "  *) exit 1 ;;\n"
      "esac\n");

  const std::string original_path = std::getenv("PATH") == nullptr ? "" : std::getenv("PATH");
  const auto current_dir = fs::current_path();
  setenv("PATH", ((temp_root / "bin").string() + ":" + original_path).c_str(), 1);
  unsetenv("RLPROF_PYTHON_EXECUTABLE");
  unsetenv("RLPROF_VLLM_EXECUTABLE");
  fs::current_path(temp_root);

  const auto environment = rlprof::inspect_runtime_environment();
  expect_true(environment.python.found, "expected python resolution to succeed");
  expect_true(!environment.python.ambiguous, "expected python resolution to avoid false ambiguity");
  expect_true(environment.python.resolved_path == (temp_root / "bin" / "python3").string(),
              "expected python3 to win over python");

  fs::current_path(current_dir);
  setenv("PATH", original_path.c_str(), 1);
  fs::remove_all(temp_root);
  return 0;
}
