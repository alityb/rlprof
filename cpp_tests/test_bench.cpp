#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "hotpath/bench/registry.h"
#include "hotpath/bench/runner.h"

namespace {

class FakeBackend final : public hotpath::bench::BenchmarkBackend {
 public:
  explicit FakeBackend(std::vector<double> durations)
      : durations_(std::move(durations)) {}

  double measure_ms(const std::function<void()>& fn) override {
    fn();
    const double value = durations_.at(index_);
    ++index_;
    return value;
  }

 private:
  std::vector<double> durations_;
  std::size_t index_ = 0;
};

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const auto shapes = hotpath::bench::parse_shapes("1x4096,64x4096");
  expect_true(shapes.size() == 2, "expected two parsed shapes");
  expect_true(shapes[0][0] == 1 && shapes[0][1] == 4096, "unexpected first shape");

  std::vector<hotpath::bench::Shape> calls;
  const hotpath::bench::KernelImpl impl = {
      .name = "fake",
      .setup = [](const hotpath::bench::Shape& shape, const std::string&) {
        return std::any(shape);
      },
      .fn = [&](std::any& state) {
        calls.push_back(std::any_cast<hotpath::bench::Shape>(state));
      },
      .dtypes = {"bf16"},
      .bytes_processed = [](const hotpath::bench::Shape& shape, const std::string&) {
        return static_cast<std::size_t>(shape[0] * shape[1] * 3 * 2);
      },
  };

  FakeBackend backend({1.0, 1.2, 0.8, 1.4});
  const auto results = hotpath::bench::benchmark_impl(
      "silu_and_mul",
      impl,
      {hotpath::bench::Shape{64, 4096}},
      "bf16",
      2,
      4,
      &backend);

  expect_true(results.size() == 1, "expected one bench result");
  expect_true(results[0].kernel == "silu_and_mul", "unexpected bench kernel");
  expect_true(results[0].implementation == "fake", "unexpected bench implementation");
  expect_true(results[0].shape[0] == 64 && results[0].shape[1] == 4096, "unexpected bench shape");
  expect_true(std::abs(results[0].avg_ms - 1.1) < 1e-9, "unexpected avg_ms");
  expect_true(std::abs(results[0].min_ms - 0.8) < 1e-9, "unexpected min_ms");
  expect_true(std::abs(results[0].p50_ms - 1.2) < 1e-9, "unexpected p50_ms");
  expect_true(std::abs(results[0].p99_ms - 1.4) < 1e-9, "unexpected p99_ms");
  expect_true(results[0].bandwidth_gb_s > 0.0, "unexpected bandwidth");
  expect_true(calls.size() == 6, "unexpected call count");

  return 0;
}
