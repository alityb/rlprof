#include "hotpath/ops.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "hotpath/artifacts.h"
#include "hotpath/store.h"

namespace hotpath {
namespace {

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
}

std::string shell_escape(const std::string& value) {
  std::string escaped = "'";
  for (char ch : value) {
    if (ch == '\'') {
      escaped += "'\\''";
    } else {
      escaped.push_back(ch);
    }
  }
  escaped += "'";
  return escaped;
}

bool compressible_artifact(const ArtifactEntry& artifact) {
  return artifact.kind == "nsys report" || artifact.kind == "nsys sqlite" ||
         artifact.kind == "nvidia-smi xml";
}

std::filesystem::path default_manifest_path(const std::filesystem::path& db_path) {
  return db_path.parent_path() / (db_path.stem().string() + "_manifest.json");
}

}  // namespace

std::string render_manifest_json(const std::filesystem::path& db_path) {
  const auto profile = load_profile(db_path);
  const auto artifacts = profile_artifacts(db_path);

  std::ostringstream out;
  out << "{\n";
  out << "  \"profile\": \"" << json_escape(db_path.string()) << "\",\n";
  out << "  \"meta\": {\n";
  bool first_meta = true;
  for (const auto& [key, value] : profile.meta) {
    if (!first_meta) {
      out << ",\n";
    }
    first_meta = false;
    out << "    \"" << json_escape(key) << "\": \"" << json_escape(value) << "\"";
  }
  out << "\n  },\n";
  out << "  \"artifacts\": [\n";
  for (std::size_t i = 0; i < artifacts.size(); ++i) {
    const auto& artifact = artifacts[i];
    out << "    {\"kind\": \"" << json_escape(artifact.kind) << "\", "
        << "\"path\": \"" << json_escape(artifact.path.string()) << "\", "
        << "\"exists\": " << (artifact.exists ? "true" : "false") << "}";
    if (i + 1 < artifacts.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

std::filesystem::path write_manifest(
    const std::filesystem::path& db_path,
    const std::filesystem::path& output_path) {
  const auto destination = output_path.empty() ? default_manifest_path(db_path) : output_path;
  if (!destination.parent_path().empty()) {
    std::filesystem::create_directories(destination.parent_path());
  }
  std::ofstream out(destination);
  out << render_manifest_json(db_path);
  return destination;
}

std::string cleanup_artifacts(
    const std::filesystem::path& dir,
    int keep_count,
    bool compress,
    bool apply_changes) {
  namespace fs = std::filesystem;
  std::vector<fs::directory_entry> db_entries;
  if (!fs::exists(dir)) {
    throw std::runtime_error("artifact directory does not exist: " + dir.string());
  }
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".db") {
      db_entries.push_back(entry);
    }
  }
  std::sort(
      db_entries.begin(),
      db_entries.end(),
      [](const auto& left, const auto& right) {
        return left.last_write_time() > right.last_write_time();
      });

  std::ostringstream out;
  out << "CLEANUP\n";
  out << "dir: " << dir.string() << "\n";
  out << "keep: " << keep_count << " latest profile dbs\n";
  out << "mode: " << (apply_changes ? "apply" : "dry-run");
  if (compress) {
    out << " + compress";
  }
  out << "\n\n";

  int acted = 0;
  for (std::size_t i = static_cast<std::size_t>(std::max(keep_count, 0)); i < db_entries.size(); ++i) {
    const auto db_path = db_entries[i].path();
    const auto artifacts = profile_artifacts(db_path);
    for (const auto& artifact : artifacts) {
      if (artifact.kind == "profile db" || !artifact.exists) {
        continue;
      }
      if (compress && compressible_artifact(artifact)) {
        const std::string action = "compress";
        out << action << " " << artifact.path.string() << "\n";
        if (apply_changes) {
          const auto command = "gzip -f " + shell_escape(artifact.path.string());
          if (std::system(command.c_str()) != 0) {
            throw std::runtime_error("failed to compress artifact: " + artifact.path.string());
          }
        }
        ++acted;
      }
    }
  }

  if (acted == 0) {
    out << "no artifacts selected\n";
  }
  return out.str();
}

}  // namespace hotpath
