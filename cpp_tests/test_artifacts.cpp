#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "rlprof/artifacts.h"
#include "rlprof/store.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "rlprof_test_artifacts";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root);

  rlprof::ProfileData profile;
  profile.meta = {
      {"model_name", "Qwen/Qwen3-8B"},
      {"artifact_nsys_rep_path", (temp_root / "sample.nsys-rep").string()},
      {"artifact_nsys_sqlite_path", (temp_root / "sample.sqlite").string()},
      {"measurement_nvidia_smi_xml", (temp_root / "sample_nvidia_smi.xml").string()},
  };
  profile.traffic_stats = {
      .total_requests = 1,
      .completion_length_mean = 1.0,
      .completion_length_p50 = 1.0,
      .completion_length_p99 = 1.0,
      .max_median_ratio = 1.0,
      .errors = 0,
  };

  const fs::path db_path = temp_root / "sample.db";
  rlprof::save_profile(db_path, profile);
  std::ofstream(temp_root / "sample.nsys-rep").put('\n');
  std::ofstream(temp_root / "sample.sqlite").put('\n');

  const auto artifacts = rlprof::profile_artifacts(db_path);
  expect_true(!artifacts.empty(), "expected artifacts");
  bool saw_db = false;
  bool saw_nsys_rep = false;
  bool saw_xml = false;
  for (const auto& artifact : artifacts) {
    if (artifact.kind == "profile db") {
      saw_db = artifact.exists;
    } else if (artifact.kind == "nsys report") {
      saw_nsys_rep = artifact.exists;
    } else if (artifact.kind == "nvidia-smi xml") {
      saw_xml = !artifact.exists;
    }
  }

  expect_true(saw_db, "expected db artifact");
  expect_true(saw_nsys_rep, "expected nsys report artifact");
  expect_true(saw_xml, "expected missing xml artifact");

  const auto rendered = rlprof::render_artifacts(db_path, artifacts);
  expect_true(rendered.find("ARTIFACTS") != std::string::npos, "expected artifacts header");
  expect_true(rendered.find("sample.nsys-rep") != std::string::npos, "expected report path");

  fs::remove_all(temp_root);
  return 0;
}
