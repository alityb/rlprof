#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "hotpath/diff.h"
#include "hotpath/store.h"

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
  const fs::path temp_dir = fs::temp_directory_path() / "hotpath_cpp_tests";
  fs::create_directories(temp_dir);
  const fs::path path_a = temp_dir / "a.db";
  const fs::path path_b = temp_dir / "b.db";
  fs::remove(path_a);
  fs::remove(path_b);

  hotpath::save_profile(
      path_a,
      hotpath::ProfileData{
          .meta = {{"model_name", "A"}},
          .kernels = {{
              .name = "gemm_a",
              .category = "gemm",
              .total_ns = 100'000'000,
              .calls = 1,
              .avg_ns = 100'000'000,
              .min_ns = 100'000'000,
              .max_ns = 100'000'000,
              .registers = 64,
              .shared_mem = 0,
          }},
          .metrics = {},
          .metrics_summary = {},
          .traffic_stats = {.total_requests = 0, .completion_length_mean = std::nullopt, .completion_length_p50 = std::nullopt, .completion_length_p99 = std::nullopt, .max_median_ratio = std::nullopt, .errors = 0},
      });

  hotpath::save_profile(
      path_b,
      hotpath::ProfileData{
          .meta = {{"model_name", "B"}},
          .kernels = {
              {
                  .name = "gemm_b",
                  .category = "gemm",
                  .total_ns = 150'000'000,
                  .calls = 1,
                  .avg_ns = 150'000'000,
                  .min_ns = 150'000'000,
                  .max_ns = 150'000'000,
                  .registers = 64,
                  .shared_mem = 0,
              },
              {
                  .name = "attention_b",
                  .category = "attention",
                  .total_ns = 20'000'000,
                  .calls = 1,
                  .avg_ns = 20'000'000,
                  .min_ns = 20'000'000,
                  .max_ns = 20'000'000,
                  .registers = 64,
                  .shared_mem = 0,
              },
          },
          .metrics = {},
          .metrics_summary = {},
          .traffic_stats = {.total_requests = 0, .completion_length_mean = std::nullopt, .completion_length_p50 = std::nullopt, .completion_length_p99 = std::nullopt, .max_median_ratio = std::nullopt, .errors = 0},
      });

  const auto deltas = hotpath::diff_profiles(path_a, path_b);
  expect_true(deltas.size() == 2, "expected two category deltas");
  expect_true(deltas[0].category == "attention", "unexpected first category");
  expect_true(deltas[0].a_ms == 0.0, "unexpected first a_ms");
  expect_true(deltas[0].b_ms == 20.0, "unexpected first b_ms");
  expect_true(deltas[0].delta_ms == 20.0, "unexpected first delta_ms");
  expect_true(!deltas[0].delta_pct.has_value(), "unexpected first delta_pct");
  expect_true(deltas[1].category == "gemm", "unexpected second category");
  expect_true(deltas[1].a_ms == 100.0, "unexpected second a_ms");
  expect_true(deltas[1].b_ms == 150.0, "unexpected second b_ms");
  expect_true(deltas[1].delta_ms == 50.0, "unexpected second delta_ms");
  expect_true(deltas[1].delta_pct.has_value() && *deltas[1].delta_pct == 50.0, "unexpected second delta_pct");

  fs::remove(path_a);
  fs::remove(path_b);
  return 0;
}
