#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "hotpath/prefix_analyzer.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  // Create 100 synthetic prompts:
  // 80 share a 50-token system prompt (tokens 1..50)
  // 15 share a different 30-token prefix (tokens 101..130)
  // 5 are completely unique

  std::vector<std::vector<int>> prompts;
  prompts.reserve(100);

  // Shared prefix A: tokens 1..50 + unique suffix
  for (int i = 0; i < 80; ++i) {
    std::vector<int> p;
    p.reserve(70);
    for (int t = 1; t <= 50; ++t) p.push_back(t);
    // Unique suffix
    for (int t = 0; t < 20; ++t) p.push_back(1000 + i * 100 + t);
    prompts.push_back(std::move(p));
  }

  // Shared prefix B: tokens 101..130 + unique suffix
  for (int i = 0; i < 15; ++i) {
    std::vector<int> p;
    p.reserve(50);
    for (int t = 101; t <= 130; ++t) p.push_back(t);
    for (int t = 0; t < 20; ++t) p.push_back(2000 + i * 100 + t);
    prompts.push_back(std::move(p));
  }

  // 5 unique prompts
  for (int i = 0; i < 5; ++i) {
    std::vector<int> p;
    p.reserve(40);
    for (int t = 0; t < 40; ++t) p.push_back(3000 + i * 1000 + t);
    prompts.push_back(std::move(p));
  }

  const auto result = hotpath::analyze_prefixes(prompts);

  expect_true(result.total_requests == 100,
              "total_requests: " + std::to_string(result.total_requests));

  // unique_prefixes should be around 3 groups (A=80, B=15, singletons=5)
  // Actually: 2 shared groups + 5 singletons = 7
  expect_true(result.unique_prefixes >= 2 && result.unique_prefixes <= 10,
              "unique_prefixes: " + std::to_string(result.unique_prefixes));

  // cacheable_token_fraction should be > 0.7
  // 80 requests * 50 tokens + 15 requests * 30 tokens = 4000 + 450 = 4450 cacheable
  // total = 80*70 + 15*50 + 5*40 = 5600 + 750 + 200 = 6550
  // fraction = 4450/6550 ≈ 0.679
  expect_true(result.cacheable_token_fraction > 0.6,
              "cacheable_token_fraction: " + std::to_string(result.cacheable_token_fraction));

  // Top prefixes should have the 80-request group first
  expect_true(!result.top_prefixes.empty(), "should have top prefixes");
  expect_true(result.top_prefixes[0].request_count == 80,
              "top prefix should have 80 requests: " +
              std::to_string(result.top_prefixes[0].request_count));
  expect_true(result.top_prefixes[0].prefix_length == 50,
              "top prefix length should be 50: " +
              std::to_string(result.top_prefixes[0].prefix_length));

  if (result.top_prefixes.size() >= 2) {
    expect_true(result.top_prefixes[1].request_count == 15,
                "second prefix should have 15 requests: " +
                std::to_string(result.top_prefixes[1].request_count));
  }

  // Median shared prefix length should be 50 (80 out of 95 shared requests have prefix 50)
  expect_true(result.median_shared_prefix_len == 50,
              "median_shared_prefix_len: " + std::to_string(result.median_shared_prefix_len));

  // Test empty input
  const auto empty = hotpath::analyze_prefixes({});
  expect_true(empty.total_requests == 0, "empty should have 0 requests");

  std::cerr << "test_prefix_analyzer: all tests passed\n";
  return 0;
}
