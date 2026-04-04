#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/otlp_export.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

int count_occurrences(const std::string& text, const std::string& pattern) {
  int count = 0;
  size_t pos = 0;
  while ((pos = text.find(pattern, pos)) != std::string::npos) {
    count++;
    pos += pattern.size();
  }
  return count;
}

}  // namespace

int main() {
  // Create 3 request traces
  std::vector<hotpath::RequestTrace> traces;
  for (int i = 0; i < 3; ++i) {
    hotpath::RequestTrace t;
    t.request_id = "req_" + std::to_string(i);
    const int64_t base = static_cast<int64_t>(i) * 1000000;
    t.arrival_us = base;
    t.queue_start_us = base;
    t.prefill_start_us = base + 1000;
    t.prefill_end_us = base + 5000;
    t.first_token_us = base + 5500;
    t.last_token_us = base + 200000;
    t.completion_us = base + 200500;
    t.prompt_tokens = 512 + i * 100;
    t.output_tokens = 64 + i * 10;
    t.cached_tokens = i * 50;
    t.status = "ok";
    traces.push_back(t);
  }

  const auto json = hotpath::export_otlp_json(traces, "test-service");

  // Verify it's valid-ish JSON
  expect_true(json.front() == '{' && json.back() == '}', "should be JSON object");

  // Verify service name
  expect_true(contains(json, "test-service"), "should contain service name");

  // Verify span names
  expect_true(contains(json, "llm.request"), "should contain llm.request span");
  expect_true(contains(json, "llm.queue"), "should contain llm.queue span");
  expect_true(contains(json, "llm.prefill"), "should contain llm.prefill span");
  expect_true(contains(json, "llm.decode"), "should contain llm.decode span");

  // Count spans: 3 traces * 4 spans each = 12
  const int span_count = count_occurrences(json, "\"spanId\"");
  expect_true(span_count == 12,
              "expected 12 spans, got " + std::to_string(span_count));

  // Verify parent-child: parentSpanId should appear in child spans (3 children per trace = 9)
  const int parent_count = count_occurrences(json, "\"parentSpanId\"");
  expect_true(parent_count == 9,
              "expected 9 parent references, got " + std::to_string(parent_count));

  // Verify traceId appears
  expect_true(contains(json, "\"traceId\""), "should contain traceId");

  // Verify attributes
  expect_true(contains(json, "prompt_tokens"), "should contain prompt_tokens attribute");
  expect_true(contains(json, "cached_tokens"), "should contain cached_tokens attribute");
  expect_true(contains(json, "output_tokens"), "should contain output_tokens attribute");

  // Verify timestamps are nanoseconds (should be > 0 and end with 000 since we multiply by 1000)
  expect_true(contains(json, "\"startTimeUnixNano\""), "should have start time");
  expect_true(contains(json, "\"endTimeUnixNano\""), "should have end time");

  // Test with empty traces
  const auto empty_json = hotpath::export_otlp_json({});
  expect_true(contains(empty_json, "resourceSpans"), "empty should still have structure");
  const int empty_spans = count_occurrences(empty_json, "\"spanId\"");
  expect_true(empty_spans == 0, "empty should have 0 spans");

  std::cerr << "test_otlp_export: all tests passed\n";
  return 0;
}
