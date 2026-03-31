#include "rlprof/doctor.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "rlprof/clock_control.h"

namespace rlprof {
namespace {

struct CommandResult {
  bool ok = false;
  std::string output;
};

CommandResult run_command(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen((command + " 2>&1").c_str(), "r");
  if (pipe == nullptr) {
    return {};
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return CommandResult{.ok = rc == 0, .output = output};
}

std::string first_line(const std::string& text) {
  const auto pos = text.find('\n');
  return pos == std::string::npos ? text : text.substr(0, pos);
}

std::string last_line(const std::string& text) {
  if (text.empty()) {
    return "";
  }
  const auto pos = text.rfind('\n');
  return pos == std::string::npos ? text : text.substr(pos + 1);
}

std::string resolved_python() {
  const char* configured = std::getenv("RLPROF_PYTHON_EXECUTABLE");
  if (configured != nullptr && std::string(configured).size() > 0) {
    return configured;
  }
  if (std::filesystem::exists(".venv/bin/python")) {
    return ".venv/bin/python";
  }
  return "python3";
}

std::string resolved_vllm() {
  const char* configured = std::getenv("RLPROF_VLLM_EXECUTABLE");
  if (configured != nullptr && std::string(configured).size() > 0) {
    return configured;
  }
  if (std::filesystem::exists(".venv/bin/vllm")) {
    return ".venv/bin/vllm";
  }
  return "vllm";
}

std::string status_label(DoctorStatus status) {
  switch (status) {
    case DoctorStatus::kPass:
      return "PASS";
    case DoctorStatus::kWarn:
      return "WARN";
    case DoctorStatus::kFail:
      return "FAIL";
  }
  return "FAIL";
}

}  // namespace

std::vector<DoctorCheck> run_doctor() {
  std::vector<DoctorCheck> checks;

  const auto nvidia_smi = run_command(
      "nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1");
  checks.push_back(DoctorCheck{
      .name = "nvidia-smi",
      .status = nvidia_smi.ok && !nvidia_smi.output.empty() ? DoctorStatus::kPass
                                                            : DoctorStatus::kFail,
      .detail = nvidia_smi.ok ? nvidia_smi.output : first_line(nvidia_smi.output),
  });

  const auto nsys = run_command("nsys --version");
  checks.push_back(DoctorCheck{
      .name = "nsys",
      .status = nsys.ok ? DoctorStatus::kPass : DoctorStatus::kFail,
      .detail = first_line(nsys.output),
  });

  const auto vllm = run_command(resolved_vllm() + " --version");
  checks.push_back(DoctorCheck{
      .name = "vllm",
      .status = vllm.ok ? DoctorStatus::kPass : DoctorStatus::kFail,
      .detail = first_line(vllm.output),
  });

  const auto bench = run_command(
      resolved_python() +
      " -c \"import torch, vllm, rlprof_py.bench_cuda; "
      "raise SystemExit(0 if torch.cuda.is_available() else 1)\"");
  checks.push_back(DoctorCheck{
      .name = "bench helper",
      .status = bench.ok ? DoctorStatus::kPass : DoctorStatus::kWarn,
      .detail = bench.ok ? "torch, vllm, and CUDA available" : last_line(bench.output),
  });

  const ClockPolicyInfo clock_policy = query_clock_policy();
  checks.push_back(DoctorCheck{
      .name = "clock policy",
      .status = !clock_policy.query_ok
                    ? DoctorStatus::kWarn
                    : (clock_policy.gpu_clocks_locked ? DoctorStatus::kPass
                                                      : DoctorStatus::kWarn),
      .detail = render_clock_policy(clock_policy),
  });

  const auto nsys_env = run_command("nsys status -e");
  checks.push_back(DoctorCheck{
      .name = "nsys environment",
      .status = nsys_env.ok ? DoctorStatus::kPass : DoctorStatus::kWarn,
      .detail = first_line(nsys_env.output),
  });

  return checks;
}

std::string render_doctor_report(const std::vector<DoctorCheck>& checks, bool color) {
  const char* reset = color ? "\033[0m" : "";
  const char* bold = color ? "\033[1m" : "";
  const char* green = color ? "\033[32m" : "";
  const char* yellow = color ? "\033[33m" : "";
  const char* red = color ? "\033[31m" : "";

  std::ostringstream out;
  out << "DOCTOR\n\n";
  out << std::left << std::setw(18) << "check" << "  "
      << std::setw(6) << "status" << "  detail\n";
  out << std::string(72, '-') << "\n";
  for (const auto& check : checks) {
    const char* status_color = reset;
    if (check.status == DoctorStatus::kPass) {
      status_color = green;
    } else if (check.status == DoctorStatus::kWarn) {
      status_color = yellow;
    } else {
      status_color = red;
    }
    out << std::left << std::setw(18) << check.name << "  "
        << status_color << bold << std::setw(6) << status_label(check.status)
        << reset << "  " << check.detail << "\n";
  }
  return out.str();
}

}  // namespace rlprof
