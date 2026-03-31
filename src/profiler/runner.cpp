#include "rlprof/profiler/runner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <sqlite3.h>
#include <errno.h>
#include <sys/types.h>

#include "rlprof/clock_control.h"
#include "rlprof/doctor.h"
#include "rlprof/profiler/parser.h"
#include "rlprof/profiler/vllm_metrics.h"
#include "rlprof/store.h"
#include "rlprof/traffic.h"

namespace rlprof::profiler {
namespace {

struct GpuTelemetrySample {
  double sample_time = 0.0;
  std::string driver_version;
  std::string pstate;
  std::string persistence_mode;
  double sm_clock_mhz = 0.0;
  double mem_clock_mhz = 0.0;
  double temp_c = 0.0;
  double power_draw_w = 0.0;
  double power_limit_w = 0.0;
  std::string throttle_active;
  std::string sw_power_cap;
  std::string sw_thermal_slowdown;
  std::string hw_thermal_slowdown;
};

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

std::string run_command(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("failed to run command: " + command);
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  const int rc = pclose(pipe);
  if (rc != 0) {
    throw std::runtime_error("command failed: " + command);
  }
  return output;
}

pid_t start_background_command(
    const std::string& command,
    const std::filesystem::path& log_path) {
  const std::string wrapped =
      "bash -lc '" + command + " > " + log_path.string() + " 2>&1 & echo $!'";
  const std::string output = run_command(wrapped);
  return static_cast<pid_t>(std::stol(output));
}

bool sqlite_kernel_table_ready(const std::filesystem::path& path) {
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    if (db != nullptr) {
      sqlite3_close(db);
    }
    return false;
  }

  sqlite3_stmt* stmt = nullptr;
  constexpr const char* kSql =
      "SELECT 1 FROM sqlite_master "
      "WHERE type = 'table' AND name = 'CUPTI_ACTIVITY_KIND_KERNEL' LIMIT 1";
  const int prepare_rc = sqlite3_prepare_v2(db, kSql, -1, &stmt, nullptr);
  if (prepare_rc != SQLITE_OK) {
    sqlite3_close(db);
    return false;
  }

  const int step_rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return step_rc == SQLITE_ROW;
}

bool wait_for_kernel_table(const std::filesystem::path& path, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::exists(path) && sqlite_kernel_table_ready(path)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return std::filesystem::exists(path) && sqlite_kernel_table_ready(path);
}

void stop_process(pid_t pid) {
  if (pid > 0) {
    kill(pid, SIGTERM);
  }
}

bool process_alive(pid_t pid) {
  if (pid <= 0) {
    return false;
  }
  const int rc = kill(pid, 0);
  if (rc == 0) {
    return true;
  }
  return errno != ESRCH;
}

bool server_ready(const std::string& server_url) {
  const std::string command =
      "curl -fsS " + server_url + "/metrics > /dev/null 2>&1";
  return std::system(command.c_str()) == 0;
}

std::string trim(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::string server_log_tail(
    const std::filesystem::path& log_path,
    std::size_t max_lines = 120) {
  std::ifstream stream(log_path);
  if (!stream.good()) {
    return "";
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(stream, line)) {
    lines.push_back(line);
  }

  if (lines.empty()) {
    return "";
  }

  const std::size_t start =
      lines.size() > max_lines ? lines.size() - max_lines : 0;
  std::ostringstream tail;
  for (std::size_t i = start; i < lines.size(); ++i) {
    tail << lines[i] << "\n";
  }
  return trim(tail.str());
}

std::runtime_error server_startup_error(
    const std::string& message,
    const std::filesystem::path& log_path) {
  const std::string tail = server_log_tail(log_path);
  if (tail.empty()) {
    return std::runtime_error(message);
  }
  return std::runtime_error(message + "\n\nvLLM log tail:\n" + tail);
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string part;
  while (std::getline(stream, part, ',')) {
    fields.push_back(trim(part));
  }
  return fields;
}

double now_seconds() {
  using Clock = std::chrono::system_clock;
  const auto now = Clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

std::optional<double> parse_double_field(const std::string& field) {
  if (field.empty() || field == "[N/A]" || field == "N/A") {
    return std::nullopt;
  }
  return std::stod(field);
}

std::optional<GpuTelemetrySample> query_gpu_telemetry() {
  try {
    const std::string output = trim(run_command(
        "nvidia-smi --query-gpu="
        "timestamp,driver_version,pstate,persistence_mode,"
        "clocks.current.sm,clocks.current.memory,temperature.gpu,"
        "power.draw,power.limit,"
        "clocks_throttle_reasons.active,"
        "clocks_throttle_reasons.sw_power_cap,"
        "clocks_throttle_reasons.sw_thermal_slowdown,"
        "clocks_throttle_reasons.hw_thermal_slowdown "
        "--format=csv,noheader,nounits | head -n 1"));
    const auto fields = split_csv_line(output);
    if (fields.size() < 13) {
      return std::nullopt;
    }
    const auto sm_clock = parse_double_field(fields[4]);
    const auto mem_clock = parse_double_field(fields[5]);
    const auto temp = parse_double_field(fields[6]);
    const auto power_draw = parse_double_field(fields[7]);
    const auto power_limit = parse_double_field(fields[8]);
    if (!sm_clock.has_value() || !mem_clock.has_value() || !temp.has_value() ||
        !power_draw.has_value() || !power_limit.has_value()) {
      return std::nullopt;
    }
    return GpuTelemetrySample{
        .sample_time = now_seconds(),
        .driver_version = fields[1],
        .pstate = fields[2],
        .persistence_mode = fields[3],
        .sm_clock_mhz = *sm_clock,
        .mem_clock_mhz = *mem_clock,
        .temp_c = *temp,
        .power_draw_w = *power_draw,
        .power_limit_w = *power_limit,
        .throttle_active = fields[9],
        .sw_power_cap = fields[10],
        .sw_thermal_slowdown = fields[11],
        .hw_thermal_slowdown = fields[12],
    };
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

void write_nvidia_smi_snapshot(const std::filesystem::path& output_prefix) {
  std::filesystem::path snapshot_path = output_prefix;
  snapshot_path += "_nvidia_smi.xml";
  const std::string command =
      "nvidia-smi -q -x > " + snapshot_path.string() + " 2>/dev/null";
  const int rc = std::system(command.c_str());
  static_cast<void>(rc);
}

std::vector<GpuTelemetrySample> poll_gpu_telemetry(
    std::chrono::milliseconds interval,
    const std::atomic<bool>& stop_flag) {
  std::vector<GpuTelemetrySample> samples;
  while (!stop_flag.load()) {
    if (const auto sample = query_gpu_telemetry(); sample.has_value()) {
      samples.push_back(*sample);
    }

    auto slept = std::chrono::milliseconds::zero();
    while (!stop_flag.load() && slept < interval) {
      constexpr auto kSlice = std::chrono::milliseconds(50);
      std::this_thread::sleep_for(kSlice);
      slept += kSlice;
    }
  }
  return samples;
}

template <typename Getter>
double max_value(const std::vector<GpuTelemetrySample>& samples, Getter getter) {
  double value = std::numeric_limits<double>::lowest();
  for (const auto& sample : samples) {
    value = std::max(value, getter(sample));
  }
  return value;
}

template <typename Getter>
double min_value(const std::vector<GpuTelemetrySample>& samples, Getter getter) {
  double value = std::numeric_limits<double>::max();
  for (const auto& sample : samples) {
    value = std::min(value, getter(sample));
  }
  return value;
}

std::map<std::string, std::string> measurement_metadata(
    const std::vector<GpuTelemetrySample>& samples) {
  std::map<std::string, std::string> metadata;
  if (samples.empty()) {
    return metadata;
  }

  std::set<std::string> pstates;
  std::size_t throttle_active_samples = 0;
  std::size_t power_cap_samples = 0;
  std::size_t thermal_samples = 0;
  for (const auto& sample : samples) {
    pstates.insert(sample.pstate);
    if (sample.throttle_active != "Not Active") {
      ++throttle_active_samples;
    }
    if (sample.sw_power_cap == "Active") {
      ++power_cap_samples;
    }
    if (sample.sw_thermal_slowdown == "Active" ||
        sample.hw_thermal_slowdown == "Active") {
      ++thermal_samples;
    }
  }

  std::ostringstream pstates_joined;
  for (auto it = pstates.begin(); it != pstates.end(); ++it) {
    if (it != pstates.begin()) {
      pstates_joined << ",";
    }
    pstates_joined << *it;
  }

  const double sm_min = min_value(samples, [](const auto& sample) { return sample.sm_clock_mhz; });
  const double sm_max = max_value(samples, [](const auto& sample) { return sample.sm_clock_mhz; });
  double sm_sum = 0.0;
  double mem_sum = 0.0;
  double power_sum = 0.0;
  double temp_min = std::numeric_limits<double>::max();
  double temp_max = std::numeric_limits<double>::lowest();
  double mem_min = std::numeric_limits<double>::max();
  double mem_max = std::numeric_limits<double>::lowest();
  double power_peak = std::numeric_limits<double>::lowest();
  double power_limit = samples.front().power_limit_w;
  for (const auto& sample : samples) {
    sm_sum += sample.sm_clock_mhz;
    mem_sum += sample.mem_clock_mhz;
    power_sum += sample.power_draw_w;
    temp_min = std::min(temp_min, sample.temp_c);
    temp_max = std::max(temp_max, sample.temp_c);
    mem_min = std::min(mem_min, sample.mem_clock_mhz);
    mem_max = std::max(mem_max, sample.mem_clock_mhz);
    power_peak = std::max(power_peak, sample.power_draw_w);
  }
  const double sm_avg = sm_sum / samples.size();
  const double mem_avg = mem_sum / samples.size();
  const double power_avg = power_sum / samples.size();
  const bool sm_clock_unstable = sm_avg > 0.0 && ((sm_max - sm_min) / sm_avg) > 0.10;

  metadata["measurement_driver_version"] = samples.front().driver_version;
  metadata["measurement_persistence_mode"] = samples.front().persistence_mode;
  metadata["measurement_pstates"] = pstates_joined.str();
  metadata["measurement_samples"] = std::to_string(samples.size());
  metadata["measurement_sm_clock_min_mhz"] = std::to_string(sm_min);
  metadata["measurement_sm_clock_avg_mhz"] = std::to_string(sm_avg);
  metadata["measurement_sm_clock_max_mhz"] = std::to_string(sm_max);
  metadata["measurement_mem_clock_min_mhz"] = std::to_string(mem_min);
  metadata["measurement_mem_clock_avg_mhz"] = std::to_string(mem_avg);
  metadata["measurement_mem_clock_max_mhz"] = std::to_string(mem_max);
  metadata["measurement_temp_min_c"] = std::to_string(temp_min);
  metadata["measurement_temp_max_c"] = std::to_string(temp_max);
  metadata["measurement_power_draw_avg_w"] = std::to_string(power_avg);
  metadata["measurement_power_draw_peak_w"] = std::to_string(power_peak);
  metadata["measurement_power_limit_w"] = std::to_string(power_limit);
  metadata["measurement_throttle_active_samples"] = std::to_string(throttle_active_samples);
  metadata["measurement_power_capped_samples"] = std::to_string(power_cap_samples);
  metadata["measurement_thermal_slowdown_samples"] = std::to_string(thermal_samples);
  metadata["warning_sm_clock_unstable"] = sm_clock_unstable ? "true" : "false";
  metadata["warning_any_clock_throttle"] = throttle_active_samples > 0 ? "true" : "false";
  metadata["warning_power_capped"] = power_cap_samples > 0 ? "true" : "false";
  metadata["warning_thermal_slowdown"] = thermal_samples > 0 ? "true" : "false";
  metadata["warning_temp_high"] = temp_max >= 80.0 ? "true" : "false";
  return metadata;
}

std::filesystem::path default_output_prefix(const ProfileConfig& config) {
  if (!config.output.empty()) {
    return config.output;
  }
  const auto now = std::chrono::system_clock::now().time_since_epoch().count();
  std::string model = config.model;
  std::replace(model.begin(), model.end(), '/', '_');
  return std::filesystem::path(".rlprof") / (model + "_" + std::to_string(now));
}

std::filesystem::path server_log_path_for_output(
    const std::filesystem::path& output_prefix) {
  return output_prefix.parent_path() /
         (output_prefix.stem().string() + "_server.log");
}

std::int64_t inferred_max_model_len(const ProfileConfig& config) {
  return std::max<std::int64_t>(2048, config.input_len + config.max_tokens);
}

std::string vllm_binary() {
  const char* configured = std::getenv("RLPROF_VLLM_EXECUTABLE");
  if (configured != nullptr && std::string(configured).size() > 0) {
    return configured;
  }
  if (std::filesystem::exists(".venv/bin/vllm")) {
    return ".venv/bin/vllm";
  }
  return "vllm";
}

bool profile_help_mentions_pid_attach(const std::string& nsys_path) {
  try {
    const std::string help = run_command(shell_escape(nsys_path) + " profile --help");
    return help.find("--pid") != std::string::npos;
  } catch (...) {
    return false;
  }
}

void validate_profile_environment(
    const ProfileConfig& config,
    const RuntimeEnvironmentInfo& environment) {
  if (config.attach_pid > 0 && config.attach_server.empty()) {
    throw std::runtime_error("--attach-pid requires --attach URL");
  }
  if (!environment.nsys.found || environment.nsys.ambiguous) {
    throw std::runtime_error("nsys unavailable: " + environment.nsys.detail);
  }
  if (environment.cuda_visible_device_count <= 0) {
    throw std::runtime_error("CUDA visibility check failed: no visible NVIDIA GPUs");
  }
  if (!environment.vllm.found || environment.vllm.ambiguous) {
    throw std::runtime_error("vllm unavailable: " + environment.vllm.detail);
  }
  if (config.attach_pid > 0 &&
      !profile_help_mentions_pid_attach(environment.nsys.resolved_path)) {
    throw std::runtime_error(
        "attach-by-process requires an nsys CLI that advertises PID attach support; "
        "installed nsys is " +
        environment.nsys.version);
  }
}

std::string make_session_name(const std::filesystem::path& output_prefix) {
  std::string session = "rlprof_";
  const std::string stem = output_prefix.filename().string();
  for (unsigned char ch : stem) {
    session.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
  }
  return session;
}

void notify_progress(const ProgressCallback& progress, const std::string& status) {
  if (progress) {
    progress(status);
  }
}

std::vector<std::string> cluster_server_urls(
    const ProfileConfig& config,
    const std::string& primary_server_url) {
  std::vector<std::string> servers;
  servers.push_back(primary_server_url);
  for (const auto& peer : config.peer_servers) {
    if (!peer.empty()) {
      servers.push_back(peer);
    }
  }
  return servers;
}

std::vector<MetricEndpoint> metric_endpoints(
    const std::vector<std::string>& server_urls) {
  std::vector<MetricEndpoint> endpoints;
  endpoints.reserve(server_urls.size());
  for (std::size_t i = 0; i < server_urls.size(); ++i) {
    endpoints.push_back(MetricEndpoint{
        .source = i == 0 ? "local" : "peer" + std::to_string(i),
        .server_url = server_urls[i],
    });
  }
  return endpoints;
}

struct MonitoringCapture {
  std::vector<MetricSample> metrics;
  std::vector<GpuTelemetrySample> gpu_samples;
};

TrafficRun run_profile_traffic(
    const ProfileConfig& config,
    const std::vector<std::string>& traffic_servers) {
  return fire_rl_traffic(
      traffic_servers,
      config.prompts,
      config.rollouts,
      config.min_tokens,
      config.max_tokens,
      config.input_len);
}

void wait_for_server_ready_or_throw(
    pid_t server_pid,
    const std::string& server_url,
    std::int64_t startup_timeout_s,
    const ProgressCallback& progress,
    const std::filesystem::path& log_path) {
  notify_progress(progress, "Waiting for server ready...");
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(startup_timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!process_alive(server_pid)) {
      throw server_startup_error("vLLM server exited before becoming ready", log_path);
    }
    if (server_ready(server_url)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  stop_process(server_pid);
  throw server_startup_error(
      "vLLM server did not become ready within " +
      std::to_string(startup_timeout_s) + " seconds",
      log_path);
}

MonitoringCapture capture_measured_run(
    const ProfileConfig& config,
    const std::vector<std::string>& traffic_servers,
    const std::vector<MetricEndpoint>& metrics_endpoints,
    const std::optional<std::string>& session_name,
    const ProgressCallback& progress,
    TrafficStats& traffic_stats) {
  const bool trace_enabled = session_name.has_value();
  if (trace_enabled) {
    notify_progress(progress, "Collecting nsys trace...");
    run_command("nsys start --session=" + *session_name);
  } else {
    notify_progress(progress, "Collecting metrics...");
  }

  std::atomic<bool> stop_flag{false};
  std::vector<MetricSample> metrics;
  std::vector<GpuTelemetrySample> gpu_samples;
  std::thread metrics_thread([&]() {
    metrics = poll_metrics(
        metrics_endpoints,
        std::chrono::milliseconds(config.metrics_interval_ms),
        stop_flag);
  });
  std::thread gpu_thread([&]() {
    gpu_samples = poll_gpu_telemetry(
        std::chrono::milliseconds(config.metrics_interval_ms),
        stop_flag);
  });

  try {
    notify_progress(
        progress,
        "Firing RL traffic (" +
            std::to_string(config.prompts * config.rollouts) + " requests)...");
    const TrafficRun traffic_run = run_profile_traffic(config, traffic_servers);
    traffic_stats = traffic_run.stats;

    stop_flag.store(true);
    if (metrics_thread.joinable()) {
      metrics_thread.join();
    }
    if (gpu_thread.joinable()) {
      gpu_thread.join();
    }
    if (trace_enabled) {
      run_command("nsys stop --session=" + *session_name);
    }
    return MonitoringCapture{
        .metrics = std::move(metrics),
        .gpu_samples = std::move(gpu_samples),
    };
  } catch (...) {
    stop_flag.store(true);
    if (metrics_thread.joinable()) {
      metrics_thread.join();
    }
    if (gpu_thread.joinable()) {
      gpu_thread.join();
    }
    if (trace_enabled) {
      run_command("nsys stop --session=" + *session_name);
    }
    throw;
  }
}

}  // namespace

ProfileRunResult run_profile(const ProfileConfig& config, ProgressCallback progress) {
  const std::filesystem::path output_prefix = default_output_prefix(config);
  const std::filesystem::path server_log_path = server_log_path_for_output(output_prefix);
  std::filesystem::path nvidia_smi_snapshot_path = output_prefix;
  nvidia_smi_snapshot_path += "_nvidia_smi.xml";
  if (!output_prefix.parent_path().empty()) {
    std::filesystem::create_directories(output_prefix.parent_path());
  }

  const bool attach_mode = !config.attach_server.empty();
  const bool attach_by_process = attach_mode && config.attach_pid > 0;
  const std::string server_url =
      attach_mode ? config.attach_server : "http://127.0.0.1:" + std::to_string(config.port);
  const std::vector<std::string> traffic_servers =
      cluster_server_urls(config, server_url);
  const std::vector<MetricEndpoint> metrics_endpoints =
      metric_endpoints(traffic_servers);
  const RuntimeEnvironmentInfo environment = inspect_runtime_environment();
  validate_profile_environment(config, environment);
  const std::string gpu_name = environment.gpu_name;
  const rlprof::ClockPolicyInfo clock_policy = rlprof::query_clock_policy();
  const std::string vllm_path =
      environment.vllm.resolved_path.empty() ? vllm_binary() : environment.vllm.resolved_path;
  const std::string vllm_version = environment.vllm.version;
  const std::int64_t max_model_len = inferred_max_model_len(config);
  const std::string session_name = make_session_name(output_prefix);
  write_nvidia_smi_snapshot(output_prefix);
  pid_t server_pid = -1;
  if (!attach_mode) {
    notify_progress(progress, "Starting vLLM server...");
    const std::string serve_command =
        "VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile --trace=cuda,nvtx,osrt "
        "--sample=none --cpuctxsw=none --export=sqlite "
        "--force-overwrite=true --session-new " + session_name +
        " --start-later=true -o " + output_prefix.string() +
        " " + vllm_path + " serve " + config.model +
        " --port " + std::to_string(config.port) +
        " --tensor-parallel-size " + std::to_string(config.tp) +
        " --max-model-len " + std::to_string(max_model_len) +
        (config.trust_remote_code ? " --trust-remote-code" : "");
    server_pid = start_background_command(serve_command, server_log_path);
  }

  if (attach_by_process) {
    throw std::runtime_error(
        "attach-by-process requested for PID " + std::to_string(config.attach_pid) +
        ", but this build only enables it when the installed nsys exposes a supported "
        "PID attach workflow. Installed nsys: " + environment.nsys.version);
  }

  try {
    if (!attach_mode) {
      wait_for_server_ready_or_throw(
          server_pid, server_url, config.startup_timeout_s, progress, server_log_path);
    } else if (!server_ready(server_url)) {
      throw std::runtime_error("attached server is not ready: " + server_url);
    }

    if (config.discard_first_run) {
      notify_progress(progress, "Warming up server...");
      static_cast<void>(run_profile_traffic(config, traffic_servers));
    }

    if (config.start_at_unix_ms > 0) {
      const auto now_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
      if (config.start_at_unix_ms > now_ms) {
        notify_progress(progress, "Waiting for synchronized start...");
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config.start_at_unix_ms - now_ms));
      }
    }

    TrafficStats traffic_stats{
        .total_requests = 0,
        .completion_length_mean = std::nullopt,
        .completion_length_p50 = std::nullopt,
        .completion_length_p99 = std::nullopt,
        .max_median_ratio = std::nullopt,
        .errors = 0,
    };
    const MonitoringCapture capture = capture_measured_run(
        config,
        traffic_servers,
        metrics_endpoints,
        attach_mode ? std::nullopt : std::optional<std::string>(session_name),
        progress,
        traffic_stats);
    if (!attach_mode) {
      stop_process(server_pid);
    }

    std::filesystem::path sqlite_path;
    std::vector<KernelRecord> kernels;
    if (!attach_mode) {
      sqlite_path = output_prefix;
      sqlite_path.replace_extension(".sqlite");
      if (!wait_for_kernel_table(sqlite_path, std::chrono::seconds(30))) {
        throw std::runtime_error("timed out waiting for nsys sqlite export readiness: " + sqlite_path.string());
      }
      notify_progress(progress, "Parsing kernel data...");
      kernels = parse_nsys_sqlite(sqlite_path);
    }
    const std::vector<MetricSummary> summaries = summarize_samples(capture.metrics);
    std::filesystem::path db_path = output_prefix;
    db_path.replace_extension(".db");
    std::filesystem::path nsys_rep_path = output_prefix;
    nsys_rep_path.replace_extension(".nsys-rep");

    ProfileData profile;
    profile.meta = {
        {"model_name", config.model},
        {"gpu_name", gpu_name},
        {"vllm_version", vllm_version},
        {"prompts", std::to_string(config.prompts)},
        {"rollouts", std::to_string(config.rollouts)},
        {"max_tokens", std::to_string(config.max_tokens)},
        {"min_tokens", std::to_string(config.min_tokens)},
        {"input_len", std::to_string(config.input_len)},
        {"cluster_mode", traffic_servers.size() > 1 ? "true" : "false"},
        {"cluster_endpoint_count", std::to_string(traffic_servers.size())},
        {"cluster_peer_endpoint_count", std::to_string(config.peer_servers.size())},
        {"cluster_trace_scope", traffic_servers.size() > 1 ? "local_server_plus_peers" : "local_server"},
        {"cluster_endpoints", [&]() {
           std::ostringstream joined;
           for (std::size_t i = 0; i < traffic_servers.size(); ++i) {
             if (i > 0) {
               joined << ",";
             }
             joined << traffic_servers[i];
           }
           return joined.str();
         }()},
        {"cluster_local_trace_endpoint", server_url},
        {"max_model_len", std::to_string(max_model_len)},
        {"trust_remote_code", config.trust_remote_code ? "true" : "false"},
        {"discard_first_run", config.discard_first_run ? "true" : "false"},
        {"measurement_untraced_warmup", config.discard_first_run ? "true" : "false"},
        {"attach_mode", attach_mode ? "true" : "false"},
        {"attach_server", attach_mode ? config.attach_server : ""},
        {"attach_pid", std::to_string(config.attach_pid)},
        {"port", std::to_string(config.port)},
        {"tp", std::to_string(config.tp)},
        {"measurement_nvidia_smi_xml", nvidia_smi_snapshot_path.string()},
        {"artifact_server_log_path", server_log_path.string()},
        {"artifact_db_path", db_path.string()},
        {"artifact_nsys_rep_path", attach_mode ? "" : nsys_rep_path.string()},
        {"artifact_nsys_sqlite_path", attach_mode ? "" : sqlite_path.string()},
        {"measurement_gpu_clock_policy", rlprof::render_clock_policy(clock_policy)},
        {"measurement_gpu_clocks_locked", clock_policy.gpu_clocks_locked ? "true" : "false"},
        {"measurement_gpu_clock_policy_query_ok", clock_policy.query_ok ? "true" : "false"},
        {"measurement_start_at_unix_ms", std::to_string(config.start_at_unix_ms)},
    };
    for (const auto& [key, value] : runtime_environment_metadata(environment)) {
      profile.meta[key] = value;
    }
    if (clock_policy.max_sm_clock_mhz.has_value()) {
      profile.meta["measurement_gpu_max_sm_clock_mhz"] =
          std::to_string(*clock_policy.max_sm_clock_mhz);
    }
    if (clock_policy.locked_sm_clock_mhz.has_value()) {
      profile.meta["measurement_gpu_locked_sm_clock_mhz"] =
          std::to_string(*clock_policy.locked_sm_clock_mhz);
    }
    if (clock_policy.query_ok && !clock_policy.gpu_clocks_locked) {
      profile.meta["warning_gpu_clocks_unlocked"] = "true";
    }
    if (attach_mode) {
      profile.meta["warning_no_kernel_trace"] = "true";
      profile.meta["cluster_trace_scope"] = "metrics_only_attach";
    }
    for (const auto& [key, value] : measurement_metadata(capture.gpu_samples)) {
      profile.meta[key] = value;
    }
    profile.kernels = kernels;
    profile.metrics = capture.metrics;
    profile.metrics_summary = summaries;
    profile.traffic_stats = traffic_stats;

    notify_progress(progress, "Saving profile...");
    save_profile(db_path, profile);

    return ProfileRunResult{
        .db_path = db_path,
        .nsys_sqlite_path = sqlite_path,
        .meta = profile.meta,
        .kernels = kernels,
        .metrics = capture.metrics,
        .traffic_stats = traffic_stats,
    };
  } catch (...) {
    if (!attach_mode) {
      const int stop_rc =
          std::system(("nsys stop --session=" + session_name + " > /dev/null 2>&1").c_str());
      static_cast<void>(stop_rc);
      stop_process(server_pid);
    }
    throw;
  }
}

}  // namespace rlprof::profiler
