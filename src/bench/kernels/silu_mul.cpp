#include "hotpath/bench/kernels/silu_mul.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hotpath/bench/registry.h"

namespace hotpath::bench::kernels {
namespace {

struct SiluMulState {
  std::int64_t batch;
  std::int64_t hidden;
  std::vector<float> input;
  std::vector<float> output;
};

SiluMulState make_state(const Shape& shape) {
  const std::int64_t batch = shape.at(0);
  const std::int64_t hidden = shape.at(1);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  SiluMulState state{
      .batch = batch,
      .hidden = hidden,
      .input = std::vector<float>(static_cast<std::size_t>(batch * hidden * 2)),
      .output = std::vector<float>(static_cast<std::size_t>(batch * hidden)),
  };
  for (float& value : state.input) {
    value = dist(rng);
  }
  return state;
}

float silu(float value) {
  return value / (1.0f + std::exp(-value));
}

void impl_reference(std::any& state_any) {
  auto& state = std::any_cast<SiluMulState&>(state_any);
  for (std::int64_t b = 0; b < state.batch; ++b) {
    const std::int64_t row_base = b * state.hidden * 2;
    const std::int64_t out_base = b * state.hidden;
    for (std::int64_t i = 0; i < state.hidden; ++i) {
      state.output[out_base + i] =
          silu(state.input[row_base + i]) * state.input[row_base + state.hidden + i];
    }
  }
}

void impl_unrolled(std::any& state_any) {
  auto& state = std::any_cast<SiluMulState&>(state_any);
  for (std::int64_t b = 0; b < state.batch; ++b) {
    const std::int64_t row_base = b * state.hidden * 2;
    const std::int64_t out_base = b * state.hidden;
    std::int64_t i = 0;
    for (; i + 3 < state.hidden; i += 4) {
      for (std::int64_t j = 0; j < 4; ++j) {
        state.output[out_base + i + j] =
            silu(state.input[row_base + i + j]) *
            state.input[row_base + state.hidden + i + j];
      }
    }
    for (; i < state.hidden; ++i) {
      state.output[out_base + i] =
          silu(state.input[row_base + i]) * state.input[row_base + state.hidden + i];
    }
  }
}

void impl_chunked(std::any& state_any) {
  auto& state = std::any_cast<SiluMulState&>(state_any);
  constexpr std::int64_t block = 128;
  for (std::int64_t b = 0; b < state.batch; ++b) {
    const std::int64_t row_base = b * state.hidden * 2;
    const std::int64_t out_base = b * state.hidden;
    for (std::int64_t base = 0; base < state.hidden; base += block) {
      const std::int64_t end = std::min(state.hidden, base + block);
      for (std::int64_t i = base; i < end; ++i) {
        state.output[out_base + i] =
            silu(state.input[row_base + i]) * state.input[row_base + state.hidden + i];
      }
    }
  }
}

std::size_t bytes_processed(const Shape& shape, const std::string& dtype) {
  return static_cast<std::size_t>(shape.at(0) * shape.at(1) * 3 * (dtype == "fp32" ? 4 : 2));
}

}  // namespace

void register_silu_mul() {
  register_kernel(
      "silu_and_mul",
      KernelImpl{
          .name = "native-reference",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_reference,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
  register_kernel(
      "silu_and_mul",
      KernelImpl{
          .name = "native-unrolled",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_unrolled,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
  register_kernel(
      "silu_and_mul",
      KernelImpl{
          .name = "native-chunked",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_chunked,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
}

}  // namespace hotpath::bench::kernels
