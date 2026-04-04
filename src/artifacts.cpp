#include "hotpath/artifacts.h"

#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "hotpath/store.h"

namespace hotpath {
namespace {

std::string meta_path_or(
    const std::map<std::string, std::string>& meta,
    const std::string& key,
    const std::filesystem::path& fallback) {
  const auto it = meta.find(key);
  if (it == meta.end() || it->second.empty()) {
    return fallback.string();
  }
  return it->second;
}

ArtifactEntry make_entry(const std::string& kind, const std::filesystem::path& path) {
  return ArtifactEntry{
      .kind = kind,
      .path = path,
      .exists = std::filesystem::exists(path),
  };
}

std::filesystem::path sibling_with_suffix(
    const std::filesystem::path& db_path,
    const std::string& suffix) {
  return db_path.parent_path() / (db_path.stem().string() + suffix);
}

std::string status_label(bool exists) {
  return exists ? "present" : "missing";
}

}  // namespace

std::vector<ArtifactEntry> profile_artifacts(const std::filesystem::path& db_path) {
  const ProfileData profile = load_profile(db_path);
  const auto& meta = profile.meta;
  std::vector<ArtifactEntry> artifacts;
  artifacts.push_back(make_entry("profile db", db_path));
  artifacts.push_back(make_entry(
      "nsys report",
      meta_path_or(meta, "artifact_nsys_rep_path", sibling_with_suffix(db_path, ".nsys-rep"))));
  artifacts.push_back(make_entry(
      "nsys sqlite",
      meta_path_or(meta, "artifact_nsys_sqlite_path", sibling_with_suffix(db_path, ".sqlite"))));
  artifacts.push_back(make_entry(
      "nvidia-smi xml",
      meta_path_or(
          meta,
          "measurement_nvidia_smi_xml",
          sibling_with_suffix(db_path, "_nvidia_smi.xml"))));
  artifacts.push_back(make_entry(
      "server log",
      meta_path_or(
          meta,
          "artifact_server_log_path",
          sibling_with_suffix(db_path, "_server.log"))));
  artifacts.push_back(make_entry("json export", sibling_with_suffix(db_path, ".json")));
  artifacts.push_back(make_entry("meta csv", sibling_with_suffix(db_path, "_meta.csv")));
  artifacts.push_back(make_entry("kernels csv", sibling_with_suffix(db_path, "_kernels.csv")));
  artifacts.push_back(make_entry(
      "vllm metrics csv", sibling_with_suffix(db_path, "_vllm_metrics.csv")));
  artifacts.push_back(make_entry(
      "metrics summary csv",
      sibling_with_suffix(db_path, "_vllm_metrics_summary.csv")));
  artifacts.push_back(make_entry(
      "traffic stats csv",
      sibling_with_suffix(db_path, "_traffic_stats.csv")));
  artifacts.push_back(make_entry(
      "warnings csv",
      sibling_with_suffix(db_path, "_warnings.csv")));
  return artifacts;
}

std::vector<ArtifactEntry> trace_artifacts(const std::filesystem::path& db_path) {
  const auto artifacts = profile_artifacts(db_path);
  std::vector<ArtifactEntry> trace_only;
  for (const auto& artifact : artifacts) {
    if (artifact.kind == "nsys report" || artifact.kind == "nsys sqlite" ||
        artifact.kind == "nvidia-smi xml" || artifact.kind == "profile db") {
      trace_only.push_back(artifact);
    }
  }
  return trace_only;
}

std::string render_artifacts(
    const std::filesystem::path& db_path,
    const std::vector<ArtifactEntry>& artifacts) {
  std::ostringstream out;
  out << "ARTIFACTS\n";
  out << "profile: " << db_path.string() << "\n\n";
  out << std::left << std::setw(18) << "kind" << "  "
      << std::setw(8) << "status" << "  path\n";
  out << std::string(72, '-') << "\n";
  for (const auto& artifact : artifacts) {
    out << std::left << std::setw(18) << artifact.kind << "  "
        << std::setw(8) << status_label(artifact.exists) << "  "
        << artifact.path.string() << "\n";
  }
  return out.str();
}

std::string render_trace_artifacts(
    const std::filesystem::path& db_path,
    const std::vector<ArtifactEntry>& artifacts,
    bool metrics_only) {
  std::ostringstream out;
  out << "TRACE ARTIFACTS\n";
  out << "profile: " << db_path.string() << "\n";
  if (metrics_only) {
    out << "trace mode: metrics-only attach\n";
  }
  out << "\n";
  out << std::left << std::setw(18) << "kind" << "  "
      << std::setw(8) << "status" << "  path\n";
  out << std::string(72, '-') << "\n";
  for (const auto& artifact : artifacts) {
    out << std::left << std::setw(18) << artifact.kind << "  "
        << std::setw(8) << status_label(artifact.exists) << "  "
        << artifact.path.string() << "\n";
  }
  return out.str();
}

}  // namespace hotpath
