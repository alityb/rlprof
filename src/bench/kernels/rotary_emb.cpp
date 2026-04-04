#include "hotpath/bench/kernels/rotary_emb.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hotpath/bench/registry.h"

namespace hotpath::bench::kernels {
namespace {

struct RotaryState {
  std::int64_t tokens;
  std::int64_t dim;
  std::vector<float> input;
  std::vector<float> output;
  std::vector<float> cos_cache;
  std::vector<float> sin_cache;
};

RotaryState make_state(const Shape& shape) {
  const std::int64_t tokens = shape.at(0);
  const std::int64_t dim = shape.at(1);
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  RotaryState state{
      .tokens = tokens,
      .dim = dim,
      .input = std::vector<float>(static_cast<std::size_t>(tokens * dim)),
      .output = std::vector<float>(static_cast<std::size_t>(tokens * dim)),
      .cos_cache = std::vector<float>(static_cast<std::size_t>(tokens * dim / 2)),
      .sin_cache = std::vector<float>(static_cast<std::size_t>(tokens * dim / 2)),
  };
  for (float& value : state.input) value = dist(rng);
  for (std::int64_t t = 0; t < tokens; ++t) {
    for (std::int64_t i = 0; i < dim / 2; ++i) {
      const float angle = static_cast<float>(t) / std::pow(10000.0f, static_cast<float>(2 * i) / dim);
      state.cos_cache[t * (dim / 2) + i] = std::cos(angle);
      state.sin_cache[t * (dim / 2) + i] = std::sin(angle);
    }
  }
  return state;
}

void impl_reference(std::any& state_any) {
  auto& state = std::any_cast<RotaryState&>(state_any);
  for (std::int64_t t = 0; t < state.tokens; ++t) {
    const std::int64_t base = t * state.dim;
    const std::int64_t trig_base = t * (state.dim / 2);
    for (std::int64_t i = 0; i < state.dim / 2; ++i) {
      const float x0 = state.input[base + (2 * i)];
      const float x1 = state.input[base + (2 * i + 1)];
      const float c = state.cos_cache[trig_base + i];
      const float s = state.sin_cache[trig_base + i];
      state.output[base + (2 * i)] = x0 * c - x1 * s;
      state.output[base + (2 * i + 1)] = x0 * s + x1 * c;
    }
  }
}

void impl_cached(std::any& state_any) {
  impl_reference(state_any);
}

void impl_chunked(std::any& state_any) {
  impl_reference(state_any);
}

std::size_t bytes_processed(const Shape& shape, const std::string& dtype) {
  return static_cast<std::size_t>(shape.at(0) * shape.at(1) * 3 * (dtype == "fp32" ? 4 : 2));
}

}  // namespace

void register_rotary_emb() {
  register_kernel(
      "rotary_embedding",
      KernelImpl{
          .name = "native-reference",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_reference,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
  register_kernel(
      "rotary_embedding",
      KernelImpl{
          .name = "native-cached",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_cached,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
  register_kernel(
      "rotary_embedding",
      KernelImpl{
          .name = "native-chunked",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_chunked,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
}

}  // namespace hotpath::bench::kernels
