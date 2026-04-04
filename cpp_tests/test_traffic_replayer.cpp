#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "hotpath/traffic_replayer.h"

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

}  // namespace

int main() {
  namespace fs = std::filesystem;

  // Find fixture
  fs::path fixture_path = fs::path(__FILE__).parent_path() / "fixtures" / "traffic.jsonl";
  if (!fs::exists(fixture_path)) {
    fixture_path = "cpp_tests/fixtures/traffic.jsonl";
  }
  expect_true(fs::exists(fixture_path), "fixture not found: " + fixture_path.string());

  const auto requests = hotpath::load_jsonl(fixture_path);
  expect_true(requests.size() == 5, "expected 5 requests, got " + std::to_string(requests.size()));

  // Verify first request
  expect_true(requests[0].prompt == "What is the capital of France?",
              "prompt mismatch: " + requests[0].prompt);
  expect_true(requests[0].max_tokens == 64,
              "max_tokens mismatch: " + std::to_string(requests[0].max_tokens));

  // Verify third request
  expect_true(requests[2].prompt == "Write a haiku about programming.",
              "prompt 3 mismatch");
  expect_true(requests[2].max_tokens == 32, "max_tokens 3 mismatch");

  // Test request body building
  const auto body = hotpath::build_request_body(requests[0], "llama-3-70b");
  expect_true(contains(body, "\"model\": \"llama-3-70b\""), "body should contain model");
  expect_true(contains(body, "\"prompt\":"), "body should contain prompt");
  expect_true(contains(body, "\"max_tokens\": 64"), "body should contain max_tokens");
  expect_true(contains(body, "\"stream\": true"), "body should contain stream");

  // Test prompt with special characters
  hotpath::ReplayRequest special;
  special.prompt = "He said \"hello\"\nNew line\\back";
  special.max_tokens = 10;
  const auto special_body = hotpath::build_request_body(special, "test");
  expect_true(contains(special_body, "\\\"hello\\\""), "should escape quotes");
  expect_true(contains(special_body, "\\n"), "should escape newlines");

  std::cerr << "test_traffic_replayer: all tests passed\n";
  return 0;
}
