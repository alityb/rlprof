#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "hotpath/log_parser.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  namespace fs = std::filesystem;

  // Find fixture file relative to executable or source tree
  fs::path fixture_path = fs::path(__FILE__).parent_path() / "fixtures" / "vllm_debug.log";
  if (!fs::exists(fixture_path)) {
    fixture_path = "cpp_tests/fixtures/vllm_debug.log";
  }
  expect_true(fs::exists(fixture_path), "fixture file not found: " + fixture_path.string());

  const auto traces = hotpath::parse_vllm_log(fixture_path);

  // Should have 5 request lifecycles
  expect_true(traces.size() == 5, "expected 5 traces, got " + std::to_string(traces.size()));

  // Verify all request IDs are present
  std::vector<std::string> ids;
  for (const auto& t : traces) {
    ids.push_back(t.request_id);
  }
  std::sort(ids.begin(), ids.end());
  expect_true(ids[0] == "cmpl-abc001", "missing cmpl-abc001");
  expect_true(ids[1] == "cmpl-abc002", "missing cmpl-abc002");
  expect_true(ids[2] == "cmpl-abc003", "missing cmpl-abc003");
  expect_true(ids[3] == "cmpl-abc004", "missing cmpl-abc004");
  expect_true(ids[4] == "cmpl-abc005", "missing cmpl-abc005");

  // Find a specific trace and check timestamps are ordered
  for (const auto& t : traces) {
    if (t.arrival_us > 0 && t.prefill_start_us > 0) {
      expect_true(t.prefill_start_us >= t.arrival_us,
                  t.request_id + ": prefill_start should be >= arrival");
    }
    if (t.prefill_start_us > 0 && t.prefill_end_us > 0) {
      expect_true(t.prefill_end_us >= t.prefill_start_us,
                  t.request_id + ": prefill_end should be >= prefill_start");
    }
    if (t.completion_us > 0 && t.arrival_us > 0) {
      expect_true(t.completion_us >= t.arrival_us,
                  t.request_id + ": completion should be >= arrival");
    }
  }

  // Verify each trace has events
  for (const auto& t : traces) {
    expect_true(!t.events.empty(), t.request_id + ": should have events");
  }

  // Test parse_vllm_log_lines with minimal input
  std::vector<std::string> lines = {
      "INFO 01-15 10:00:00.000000 api_server.py:100 Added request cmpl-test001 with prompt_token=64",
      "INFO 01-15 10:00:00.500000 engine.py:400 Finished request cmpl-test001 output_token=16",
  };
  const auto simple = hotpath::parse_vllm_log_lines(lines);
  expect_true(simple.size() == 1, "expected 1 trace from simple input");
  expect_true(simple[0].request_id == "cmpl-test001", "simple trace id mismatch");
  expect_true(simple[0].prompt_tokens == 64, "simple prompt_tokens mismatch");
  expect_true(simple[0].output_tokens == 16, "simple output_tokens mismatch");
  expect_true(simple[0].status == "ok", "simple status should be ok");

  std::cerr << "test_log_parser: all tests passed\n";
  return 0;
}
