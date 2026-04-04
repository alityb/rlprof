#include "hotpath/serve_profiler.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

#include "hotpath/batch_analyzer.h"
#include "hotpath/cache_analyzer.h"
#include "hotpath/disagg_model.h"
#include "hotpath/log_parser.h"
#include "hotpath/phase_analyzer.h"
#include "hotpath/prefix_analyzer.h"
#include "hotpath/recommender.h"
#include "hotpath/store.h"
#include "hotpath/workload_classifier.h"

namespace hotpath {

int run_serve_profile(const ServeProfileOptions& opts) {
  namespace fs = std::filesystem;

  std::cerr << "serve-profile: endpoint=" << opts.endpoint
            << " duration=" << opts.duration_seconds << "s"
            << " engine=" << opts.engine << "\n";

  // Validate endpoint is reachable
  const std::string curl_cmd = "curl -fsS --max-time 5 " + opts.endpoint + "/health 2>/dev/null";
  const int curl_rc = std::system(curl_cmd.c_str());
  if (curl_rc != 0) {
    std::cerr << "warning: endpoint " << opts.endpoint << " may not be reachable\n";
  }

  // Create output directory
  const fs::path output_dir(opts.output);
  fs::create_directories(output_dir);
  const fs::path db_path = output_dir / "serve_profile.db";

  // Initialize database
  init_db(db_path);

  std::cerr << "serve-profile: collecting metrics for " << opts.duration_seconds << " seconds...\n";

  // In a real implementation, this would:
  // 1. Start polling /metrics at 1Hz
  // 2. Start capturing logs
  // 3. Optionally start nsys session
  // 4. Replay traffic from JSONL
  // 5. After duration, stop all collection
  // 6. Parse logs -> request traces
  // 7. Categorize kernels with phase info
  // 8. Run analyzers
  // 9. Run workload classifier + disagg model + recommender
  // 10. Store everything in SQLite

  // For now, this is a structural placeholder that compiles and links
  // all the serving modules together.

  std::cerr << "serve-profile: profiling requires a live " << opts.engine << " server.\n";
  std::cerr << "serve-profile: results will be stored in " << db_path.string() << "\n";

  return 0;
}

}  // namespace hotpath
