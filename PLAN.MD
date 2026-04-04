# rlprof v2 — Codex Task Plan

## Overview

Broaden rlprof from an RL-rollout profiler into a full LLM inference profiler with serving analysis, prefill/decode phase decomposition, KV cache diagnostics, and a disaggregation advisor.

Each task below is designed to be executed as **one Codex thread**. Tasks are ordered by dependency. Each specifies exactly which files to create/modify, what the inputs and outputs are, and how to verify.

---

## TASK 01 — Request lifecycle data model

**Goal:** Define the SQLite schema and C++ types for storing per-request lifecycle events.

**Create:**
- `include/rlprof/request_trace.h`
- `cpp_tests/test_request_trace.cpp`

**Modify:**
- `src/store.cpp` — add `request_traces` and `request_events` tables
- `include/rlprof/store.h` — add insert/query functions for request traces
- `CMakeLists.txt` — add test executable

**Schema:**
```sql
CREATE TABLE request_traces (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    arrival_us INTEGER,          -- microsecond timestamp
    queue_start_us INTEGER,
    prefill_start_us INTEGER,
    prefill_end_us INTEGER,
    first_token_us INTEGER,
    last_token_us INTEGER,
    completion_us INTEGER,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,       -- prefix cache hits
    status TEXT                   -- ok | preempted | error
);

CREATE TABLE request_events (
    id INTEGER PRIMARY KEY,
    trace_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,     -- queue | schedule | prefill | decode | cache_hit | preempt
    timestamp_us INTEGER NOT NULL,
    detail TEXT,                  -- JSON blob for event-specific data
    FOREIGN KEY (trace_id) REFERENCES request_traces(id)
);
```

**Types in `request_trace.h`:**
```cpp
struct RequestEvent {
    std::string event_type;
    int64_t timestamp_us;
    std::string detail;  // JSON
};

struct RequestTrace {
    std::string request_id;
    int64_t arrival_us = 0;
    int64_t queue_start_us = 0;
    int64_t prefill_start_us = 0;
    int64_t prefill_end_us = 0;
    int64_t first_token_us = 0;
    int64_t last_token_us = 0;
    int64_t completion_us = 0;
    int prompt_tokens = 0;
    int output_tokens = 0;
    int cached_tokens = 0;
    std::string status;
    std::vector<RequestEvent> events;
};
```

**Test:** Insert 100 synthetic traces, query back, verify roundtrip. Query traces where prefill > 50ms. Query traces with cached_tokens > 0.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_request_trace --output-on-failure`

---

## TASK 02 — vLLM log parser for request events

**Goal:** Parse vLLM's structured debug logs to extract per-request lifecycle timestamps.

**Create:**
- `src/serving/log_parser.cpp`
- `src/serving/log_parser.h` (or `include/rlprof/log_parser.h`)
- `cpp_tests/test_log_parser.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Behavior:**

vLLM (with `VLLM_LOGGING_LEVEL=DEBUG`) emits log lines like:
```
INFO ... Added request <req_id> ...
INFO ... Running: ... requests ...
DEBUG ... Prefill ... tokens ...
INFO ... Finished request <req_id> ...
```

The parser should:
1. Accept a log file path or a stream of log lines
2. Extract request IDs, timestamps, event types
3. Return `std::vector<RequestTrace>` with populated fields
4. Handle missing fields gracefully (not all log lines present in all vLLM versions)

**Test:** Create a fixture file `cpp_tests/fixtures/vllm_debug.log` with 5 representative request lifecycles. Parse it. Verify extracted request count, IDs, and timestamp ordering.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_log_parser --output-on-failure`

---

## TASK 03 — Phase categorizer extension

**Goal:** Extend the existing `categorizer.cpp` to classify CUDA kernels as prefill vs decode phase.

**Modify:**
- `src/profiler/categorizer.cpp` — add phase classification
- `include/rlprof/categorizer.h` — add `KernelPhase` enum and phase field to existing kernel struct
- `cpp_tests/test_categorizer.cpp` — add phase classification tests

**Logic:**

Add a `KernelPhase` enum: `{ PREFILL, DECODE, UNKNOWN }`.

Classification rules (in priority order):
1. Kernel name contains `flash_fwd` or `flash_attn_forward` → `PREFILL`
2. Kernel name contains `paged_attention` or `reshape_and_cache` → `DECODE`
3. Kernel name contains `rotary` or `rms_norm` → classify by grid size heuristic:
   - grid.x * grid.y * grid.z > threshold → `PREFILL`
   - else → `DECODE`
4. Default → `UNKNOWN`

The threshold should be configurable (default 1024).

**Test:** Existing categorizer tests must still pass. Add new tests with fixture kernel names that verify phase classification for flash_fwd, paged_attention, rms_norm with various grid sizes.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_categorizer --output-on-failure`

---

## TASK 04 — Phase analyzer: aggregate prefill/decode GPU time

**Goal:** Consume categorized kernel traces and produce phase-level time breakdown.

**Create:**
- `src/serving/phase_analyzer.cpp`
- `include/rlprof/phase_analyzer.h`
- `cpp_tests/test_phase_analyzer.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Input:** A vector of categorized kernel entries (from the parser + categorizer), each with: name, phase (PREFILL/DECODE/UNKNOWN), start_us, duration_us.

**Output:**
```cpp
struct PhaseBreakdown {
    int64_t prefill_us;       // total GPU time in prefill kernels
    int64_t decode_us;        // total GPU time in decode kernels
    int64_t unknown_us;       // unclassified
    int64_t total_us;
    double prefill_fraction;  // prefill_us / total_us
    double decode_fraction;
    int prefill_kernel_count;
    int decode_kernel_count;
};
```

Also compute per-second time series: for each 1-second window, what fraction of GPU time was prefill vs decode.

**Test:** Create 200 synthetic kernel entries (mix of prefill and decode), run analyzer, verify totals sum correctly, verify fractions are in [0, 1], verify time series has correct window count.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_phase_analyzer --output-on-failure`

---

## TASK 05 — Batch composition analyzer

**Goal:** Analyze scheduler batching behavior from vLLM metrics snapshots.

**Create:**
- `src/serving/batch_analyzer.cpp`
- `include/rlprof/batch_analyzer.h`
- `cpp_tests/test_batch_analyzer.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Input:** Time series of vLLM `/metrics` snapshots (already collected by `vllm_metrics.cpp`). Relevant metrics:
- `vllm:num_requests_running` — current batch size
- `vllm:num_requests_waiting` — queue depth
- `vllm:num_preemption_total` — preemptions (counter)
- `vllm:gpu_cache_usage_perc` — KV cache pressure

**Output:**
```cpp
struct BatchAnalysis {
    double avg_batch_size;
    double p50_batch_size;
    double p99_batch_size;
    double avg_queue_depth;
    double p99_queue_depth;
    int total_preemptions;
    double avg_cache_usage;
    double peak_cache_usage;
    // per-second time series of batch size and queue depth
    std::vector<std::pair<int64_t, double>> batch_size_series;
    std::vector<std::pair<int64_t, double>> queue_depth_series;
};
```

**Test:** Create 60 synthetic metric snapshots (1 per second, 60 seconds). Vary batch size from 1 to 32 and back. Verify percentiles, averages, series length.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_batch_analyzer --output-on-failure`

---

## TASK 06 — Cache efficiency analyzer

**Goal:** Compute KV cache efficiency metrics from metrics snapshots and request traces.

**Create:**
- `src/serving/cache_analyzer.cpp`
- `include/rlprof/cache_analyzer.h`
- `cpp_tests/test_cache_analyzer.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Input:**
- vLLM metrics time series (gpu_cache_usage_perc, cpu_cache_usage_perc)
- Request traces with `cached_tokens` and `prompt_tokens` fields

**Output:**
```cpp
struct CacheAnalysis {
    double cache_hit_rate;        // sum(cached_tokens) / sum(prompt_tokens)
    double avg_cache_usage;
    double peak_cache_usage;
    double cache_pressure_seconds; // seconds where cache > 90%
    int eviction_count;           // from preemption counter
    // histogram: what fraction of requests had 0%, 1-25%, 25-50%, 50-75%, 75-100% cache hit
    std::array<int, 5> hit_rate_histogram;
};
```

**Test:** Create 50 request traces with varying cached_tokens. Create 60 metrics snapshots with varying cache usage. Verify hit rate calculation, histogram bucket assignment, pressure seconds.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_cache_analyzer --output-on-failure`

---

## TASK 07 — Prefix sharing analyzer

**Goal:** Analyze traffic to detect prefix sharing patterns and estimate caching potential.

**Create:**
- `src/serving/prefix_analyzer.cpp`
- `include/rlprof/prefix_analyzer.h`
- `cpp_tests/test_prefix_analyzer.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Input:** A list of tokenized prompts (represented as `std::vector<std::vector<int>>`).

**Output:**
```cpp
struct PrefixAnalysis {
    int total_requests;
    int unique_prefixes;          // distinct prefixes at min_prefix_len
    double avg_requests_per_prefix;
    int median_shared_prefix_len; // in tokens
    double cacheable_token_fraction; // tokens that appear in a shared prefix / total prompt tokens
    // top-10 most common prefixes with their frequency and length
    std::vector<PrefixGroup> top_prefixes;
};

struct PrefixGroup {
    int prefix_length;  // tokens
    int request_count;
    double fraction;    // of total requests
};
```

**Algorithm:** Build a trie of token sequences. Walk the trie to find branching points. A "shared prefix" is any path from root to a branching point that has ≥2 leaf descendants.

**Test:** Create 100 synthetic prompts where 80 share a 50-token system prompt, 15 share a different 30-token prefix, and 5 are unique. Verify: unique_prefixes ≈ 3, cacheable_token_fraction > 0.7.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_prefix_analyzer --output-on-failure`

---

## TASK 08 — Workload classifier

**Goal:** Characterize a profiled workload into a class that determines disaggregation advice.

**Create:**
- `src/advisor/workload_classifier.cpp`
- `include/rlprof/workload_classifier.h`
- `cpp_tests/test_workload_classifier.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Input:** `PhaseBreakdown`, `BatchAnalysis`, `CacheAnalysis`, `PrefixAnalysis`, plus summary stats from request traces (median prompt length, median output length, request rate).

**Output:**
```cpp
enum class WorkloadClass {
    PREFILL_HEAVY,    // median prompt > 2048, output < prompt/4
    DECODE_HEAVY,     // median output > prompt * 2 (reasoning workloads)
    BALANCED,         // neither dominates
    CACHE_FRIENDLY,   // prefix sharing > 60%
    SHORT_CONTEXT,    // median prompt < 256
};

struct WorkloadProfile {
    WorkloadClass primary_class;
    double prefill_fraction;      // of GPU time
    double median_prompt_tokens;
    double median_output_tokens;
    double request_rate;          // req/s
    double prefix_sharing_rate;
    double cache_hit_rate;
    double prefill_contention;    // estimated: how much prefill slows decode p99
};
```

**Contention estimation:** `prefill_contention = (p99_decode_latency - median_decode_latency) / median_decode_latency`. If prefill is causing stalls, p99 will be much higher than median.

**Test:** Create 5 synthetic profiles representing each workload class. Verify classification is correct for each.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_workload_classifier --output-on-failure`

---

## TASK 09 — Disaggregation model

**Goal:** Analytical model that estimates throughput/latency under monolithic vs disaggregated serving.

**Create:**
- `src/advisor/disagg_model.cpp`
- `include/rlprof/disagg_model.h`
- `cpp_tests/test_disagg_model.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Input:**
```cpp
struct DisaggModelInput {
    WorkloadProfile profile;
    int total_gpus;
    double network_bandwidth_gbps;   // for KV transfer
    double avg_kv_transfer_bytes;    // per request
};
```

**Output:**
```cpp
struct DisaggEstimate {
    // monolithic baseline
    double mono_throughput_rps;
    double mono_p99_ttft_ms;
    double mono_p99_itl_ms;

    // disaggregated projection
    int optimal_prefill_gpus;
    int optimal_decode_gpus;
    double disagg_throughput_rps;
    double disagg_p99_ttft_ms;
    double disagg_p99_itl_ms;
    double kv_transfer_overhead_ms;

    // recommendation
    bool should_disaggregate;
    std::string reason;              // human-readable explanation
    double throughput_improvement;   // ratio
    double min_bandwidth_gbps;       // below this, disagg hurts
};
```

**Model (simplified M/G/1 approximation):**
- Monolithic: prefill and decode share one queue. Service time = prefill_time + decode_steps * decode_time. Prefill arrivals cause head-of-line blocking.
- Disaggregated: separate queues. Prefill queue has service time = prefill_time. Decode queue has service time = decode_time. KV transfer adds fixed latency.
- Optimal split: search P:D ratios from 1:(N-1) to (N-1):1, pick ratio that maximizes throughput subject to latency SLOs.

**Test:** 
1. Short-context workload (256 prompt, 64 output, 8 GPUs, 100Gbps) → should_disaggregate = false.
2. Long-context workload (4096 prompt, 256 output, 8 GPUs, 100Gbps) → should_disaggregate = true, optimal split ≈ 2:6.
3. Low-bandwidth scenario (4096 prompt, 256 output, 8 GPUs, 10Gbps) → should_disaggregate = false (KV transfer too slow).

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_disagg_model --output-on-failure`

---

## TASK 10 — Recommender: generate actionable config

**Goal:** Take disaggregation model output and generate ready-to-use configs for vLLM, llm-d, and Dynamo.

**Create:**
- `src/advisor/recommender.cpp`
- `include/rlprof/recommender.h`
- `cpp_tests/test_recommender.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Input:** `DisaggEstimate` + model name + engine type (vllm/sglang)

**Output:** `std::string` containing the config. Three format functions:

1. `generate_vllm_config(estimate, model)` → shell script with two `vllm serve` commands (prefill instance + decode instance) and the KV connector config JSON
2. `generate_llmd_config(estimate, model)` → YAML for llm-d Helm values (prefill replicas, decode replicas, model name)
3. `generate_dynamo_config(estimate, model)` → YAML for Dynamo deployment spec

Also generate a human-readable summary string.

**Test:** Generate configs for a "should disaggregate" and "should not disaggregate" case. Verify configs are valid (parseable YAML/JSON). Verify the "should not" case outputs a monolithic config instead.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_recommender --output-on-failure`

---

## TASK 11 — `serve-profile` CLI command

**Goal:** Wire up the new serving modules into a CLI command that profiles a live vLLM server.

**Modify:**
- `src/main.cpp` — add `serve-profile` and `serve-report` subcommand dispatch
- `src/interactive.cpp` — add interactive flow for serve-profile if applicable

**Create:**
- `src/serving/serve_profiler.cpp` — orchestrates: start metrics collection + log capture + optional nsys trace, run traffic, stop, parse logs, correlate, store
- `include/rlprof/serve_profiler.h`

**CLI signature:**
```
rlprof serve-profile \
  --endpoint http://localhost:8000 \
  --duration 60 \
  --traffic <path-to-jsonl-or-sharegpt> \
  --output .rlprof/serve_run \
  [--nsys]                          # optional: also capture kernel trace
  [--engine vllm|sglang]            # default: vllm
```

**Behavior:**
1. Verify endpoint is reachable (use existing doctor/validate infra)
2. Start polling `/metrics` at 1Hz (existing vllm_metrics module)
3. Start capturing vLLM logs (if --nsys, also start nsys session via existing runner)
4. Replay traffic from JSONL file (send requests via HTTP, record response times)
5. After duration, stop all collection
6. Parse logs → request traces (Task 02)
7. If nsys data: parse + categorize kernels with phase info (Task 03)
8. Run all analyzers (Tasks 04-07)
9. Run workload classifier + disagg model + recommender (Tasks 08-10)
10. Store everything in SQLite

**No test for this task** — it requires a live vLLM server. Verification is structural: does it compile and link.

**Verify:** `cmake --build build --parallel`

---

## TASK 12 — `serve-report` CLI command

**Goal:** Generate a human-readable report from a serve-profile database.

**Modify:**
- `src/report.cpp` — add `serve_report()` function
- `include/rlprof/report.h` — declare it
- `src/main.cpp` — wire up `serve-report` subcommand

**Create:**
- `cpp_tests/test_serve_report.cpp`

**Output format:** (printed to stdout)
```
rlprof serve-report — <model> — <engine> — <gpu_info>
══════════════════════════════════════════════════════

Requests: 847  |  Duration: 60.0s  |  Throughput: 14.1 req/s

Latency (ms)              p50      p90      p99
────────────────────────────────────────────────
Queue wait                2.1      8.4     41.2
Prefill                  12.3     45.1    112.8
Decode (total)          184.2    412.5    891.3
Decode (per-token)        4.2      5.1      8.9
End-to-end              201.4    458.3    983.1

GPU Phase Breakdown
────────────────────────────────────────────────
Prefill compute          31.2%   ████████░░░░░░
Decode compute           48.1%   █████████████░
Other / idle             20.7%   █████░░░░░░░░░

KV Cache
────────────────────────────────────────────────
Hit rate                 41.2%
Avg usage                73.2%
Evictions                12

Prefix Sharing
────────────────────────────────────────────────
Unique prefixes          23
Cacheable tokens         72.3%

Disaggregation Advisor
────────────────────────────────────────────────
Recommendation:          DISAGGREGATE
Optimal P:D ratio:       1:3
Projected throughput:    +40% (19.8 req/s)
Projected p99 TTFT:      89ms → 52ms
Min network bandwidth:   50 Gbps

Run `rlprof disagg-config .rlprof/serve_run.db` for deployment configs.
```

**Test:** Create an in-memory SQLite DB, populate with synthetic data for all tables (request_traces, metrics snapshots, phase breakdown, cache analysis, disagg estimate). Call `serve_report()`. Verify output contains expected sections (check for substring presence of "Latency", "GPU Phase", "KV Cache", "Disaggregation").

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_serve_report --output-on-failure`

---

## TASK 13 — `disagg-config` CLI command

**Goal:** Standalone command that reads a serve-profile DB and prints deployment configs.

**Modify:**
- `src/main.cpp` — add `disagg-config` subcommand

**CLI signature:**
```
rlprof disagg-config .rlprof/serve_run.db [--format vllm|llmd|dynamo|all]
```

**Behavior:** Load the `DisaggEstimate` from the DB, call the appropriate `generate_*_config()` from Task 10, print to stdout.

**No new test** — just wiring. Verify by compilation.

**Verify:** `cmake --build build --parallel`

---

## TASK 14 — Traffic replayer

**Goal:** A module that sends HTTP requests to a vLLM/SGLang endpoint from a JSONL file or ShareGPT dataset.

**Create:**
- `src/serving/traffic_replayer.cpp`
- `include/rlprof/traffic_replayer.h`
- `cpp_tests/test_traffic_replayer.cpp`

**Behavior:**
- Read a JSONL file where each line is `{"prompt": "...", "max_tokens": N}`
- Or read ShareGPT format: `{"conversations": [{"from": "human", "value": "..."}]}`
- Send requests to the OpenAI-compatible `/v1/completions` endpoint via HTTP
- Record per-request: send timestamp, first-token timestamp, completion timestamp, tokens generated
- Support configurable concurrency (max outstanding requests) and rate limiting (req/s)

Use POSIX sockets or `libcurl` if available; otherwise shell out to `curl` as a fallback. Keep it simple — this is a load generator, not a benchmarking framework.

**Test:** Parse a 5-line JSONL fixture. Verify request objects are correctly constructed. (Don't test actual HTTP — that needs a server.)

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_traffic_replayer --output-on-failure`

---

## TASK 15 — SGLang metrics parser

**Goal:** Add SGLang support alongside vLLM for metrics collection.

**Create:**
- `src/profiler/sglang_metrics.cpp`
- `include/rlprof/sglang_metrics.h`
- `cpp_tests/test_sglang_metrics.cpp`

**Modify:**
- `CMakeLists.txt` — add source and test

**Behavior:** SGLang exposes Prometheus metrics at `/metrics` with metric names like:
- `sglang:num_running_req`
- `sglang:num_waiting_req`
- `sglang:token_usage`
- `sglang:cache_hit_rate`
- `sglang:num_total_tokens`

Parse these into the same internal `MetricSnapshot` struct used by `vllm_metrics.cpp`. Create an abstract base or a tagged union so downstream code doesn't care which engine produced the metrics.

**Test:** Create a fixture string with SGLang Prometheus output. Parse it. Verify extracted values match expected.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_sglang_metrics --output-on-failure`

---

## TASK 16 — OpenTelemetry trace export

**Goal:** Export request lifecycle traces as OpenTelemetry-compatible JSON (OTLP JSON format).

**Create:**
- `src/export/otlp_export.cpp`
- `include/rlprof/otlp_export.h`
- `cpp_tests/test_otlp_export.cpp`

**Modify:**
- `src/export.cpp` — add otlp format to export dispatch
- `CMakeLists.txt` — add source and test

**Behavior:** Convert request traces into OTLP JSON spans:
- Root span: `llm.request` (arrival → completion)
  - Child: `llm.queue` (arrival → prefill_start)
  - Child: `llm.prefill` (prefill_start → prefill_end), attributes: prompt_tokens, cached_tokens
  - Child: `llm.decode` (first_token → last_token), attributes: output_tokens

Write to a JSON file that can be imported into Jaeger or Grafana Tempo.

**Test:** Create 3 request traces. Export to OTLP JSON string. Parse the JSON back. Verify span count = 3 * 4 = 12. Verify parent-child relationships. Verify timestamps are monotonic within each trace.

**Verify:** `cmake --build build --parallel && ctest --test-dir build -R test_otlp_export --output-on-failure`

---

## TASK 17 — Update README

**Goal:** Rewrite README.md to reflect v2 scope.

**Modify:**
- `README.md`

**New tagline:** "rlprof — the profiler for LLM inference. Kernel timing, request lifecycle tracing, and disaggregation analysis for vLLM and SGLang."

**Sections:**
1. What it does (3 bullets: profile, analyze, advise)
2. Quick start: `rlprof serve-profile` example
3. Commands: serve-profile, serve-report, disagg-config, profile (existing), report (existing), diff (existing), bench (existing)
4. Install: pip install rlprof
5. System requirements
6. Build from source
7. How it works (brief architecture)

**No test.** Verify by reading.

---

## Dependency Graph

```
01 (schema)
├── 02 (log parser) ──────────┐
├── 03 (phase categorizer) ───┤
│                             ├── 11 (serve-profile) ── 13 (disagg-config)
├── 04 (phase analyzer) ──────┤
├── 05 (batch analyzer) ──────┤
├── 06 (cache analyzer) ──────┤
├── 07 (prefix analyzer) ─────┤
│                             │
├── 08 (workload classifier) ─┤
├── 09 (disagg model) ────────┤
├── 10 (recommender) ─────────┘
│
├── 12 (serve-report) ← needs all analysis modules
├── 14 (traffic replayer) ← independent
├── 15 (SGLang metrics) ← independent
├── 16 (OTLP export) ← needs 01
└── 17 (README) ← last
```

**Parallelizable groups:**
- Tasks 02, 03, 04, 05, 06, 07 can all run in parallel after Task 01
- Tasks 08, 09, 10 can run in parallel after Tasks 04-07
- Tasks 14, 15, 16 are fully independent of each other
- Task 11 needs 02-10
- Task 12 needs 11
- Task 17 is last
