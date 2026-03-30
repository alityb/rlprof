#include "rlprof/bench/kernels/rms_norm.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "rlprof/bench/registry.h"

namespace rlprof::bench::kernels {
namespace {

struct RmsNormState {
  std::int64_t batch;
  std::int64_t hidden;
  std::vector<float> input;
  std::vector<float> residual;
  std::vector<float> weight;
  std::vector<float> output;
};

RmsNormState make_state(const Shape& shape) {
  const std::int64_t batch = shape.at(0);
  const std::int64_t hidden = shape.at(1);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  RmsNormState state{
      .batch = batch,
      .hidden = hidden,
      .input = std::vector<float>(static_cast<std::size_t>(batch * hidden)),
      .residual = std::vector<float>(static_cast<std::size_t>(batch * hidden)),
      .weight = std::vector<float>(static_cast<std::size_t>(hidden)),
      .output = std::vector<float>(static_cast<std::size_t>(batch * hidden)),
  };
  for (float& value : state.input) value = dist(rng);
  for (float& value : state.residual) value = dist(rng);
  for (float& value : state.weight) value = dist(rng);
  return state;
}

void impl_reference(std::any& state_any) {
  auto& state = std::any_cast<RmsNormState&>(state_any);
  constexpr float eps = 1e-5f;
  for (std::int64_t b = 0; b < state.batch; ++b) {
    const std::int64_t base = b * state.hidden;
    float sq_sum = 0.0f;
    for (std::int64_t i = 0; i < state.hidden; ++i) {
      const float value = state.input[base + i] + state.residual[base + i];
      sq_sum += value * value;
    }
    const float inv_rms = 1.0f / std::sqrt((sq_sum / state.hidden) + eps);
    for (std::int64_t i = 0; i < state.hidden; ++i) {
      const float value = state.input[base + i] + state.residual[base + i];
      state.output[base + i] = value * inv_rms * state.weight[i];
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

void register_rms_norm() {
  register_kernel(
      "fused_add_rms_norm",
      KernelImpl{
          .name = "native-reference",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_reference,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
  register_kernel(
      "fused_add_rms_norm",
      KernelImpl{
          .name = "native-cached",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_cached,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
  register_kernel(
      "fused_add_rms_norm",
      KernelImpl{
          .name = "native-chunked",
          .setup = [](const Shape& shape, const std::string&) { return std::any(make_state(shape)); },
          .fn = impl_chunked,
          .dtypes = {"bf16", "fp16", "fp32"},
          .bytes_processed = bytes_processed,
      });
}

}  // namespace rlprof::bench::kernels
