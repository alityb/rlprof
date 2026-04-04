#include <cstdlib>
#include <iostream>
#include <string_view>

#include "hotpath/profiler/categorizer.h"

namespace {

void expect_equal(std::string_view actual, std::string_view expected) {
  if (actual != expected) {
    std::cerr << "expected [" << expected << "] but got [" << actual << "]\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  using hotpath::profiler::categorize;

  expect_equal(
      categorize("sm80_xmma_gemm_bf16_bf16_bf16f32_f32_tn_128x128"),
      "gemm");
  expect_equal(
      categorize("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn"),
      "gemm");
  expect_equal(
      categorize("vllm::act_and_mul_kernel<bf16,silu,true>"),
      "activation");
  expect_equal(
      categorize("_selective_scan_update_kernel"),
      "mamba");
  expect_equal(
      categorize("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2"),
      "other");
  expect_equal(categorize("_topk_topp_kernel"), "sampling");
  expect_equal(categorize("flash_fwd_splitkv_bf16_sm80"), "attention");
  expect_equal(categorize("vectorized_elementwise_kernel"), "other");
  expect_equal(categorize("triton_poi_fused_4"), "other");
  expect_equal(categorize("some_unknown_kernel"), "other");

  // Phase classification tests
  using hotpath::profiler::classify_phase;
  using hotpath::profiler::KernelPhase;
  using hotpath::profiler::GridDim;

  // Rule 1: flash_fwd → PREFILL
  if (classify_phase("flash_fwd_splitkv_bf16_sm80") != KernelPhase::PREFILL) {
    std::cerr << "flash_fwd should be PREFILL\n";
    return 1;
  }
  if (classify_phase("flash_attn_forward_kernel") != KernelPhase::PREFILL) {
    std::cerr << "flash_attn_forward should be PREFILL\n";
    return 1;
  }

  // Rule 2: paged_attention → DECODE
  if (classify_phase("paged_attention_v2_kernel") != KernelPhase::DECODE) {
    std::cerr << "paged_attention should be DECODE\n";
    return 1;
  }
  if (classify_phase("reshape_and_cache_flash") != KernelPhase::DECODE) {
    std::cerr << "reshape_and_cache should be DECODE\n";
    return 1;
  }

  // Rule 3: rms_norm with large grid → PREFILL
  if (classify_phase("rms_norm_kernel", GridDim{2048, 1, 1}) != KernelPhase::PREFILL) {
    std::cerr << "rms_norm large grid should be PREFILL\n";
    return 1;
  }
  // rms_norm with small grid → DECODE
  if (classify_phase("rms_norm_kernel", GridDim{32, 1, 1}) != KernelPhase::DECODE) {
    std::cerr << "rms_norm small grid should be DECODE\n";
    return 1;
  }
  // rotary with large grid → PREFILL
  if (classify_phase("rotary_embedding_kernel", GridDim{64, 32, 1}) != KernelPhase::PREFILL) {
    std::cerr << "rotary large grid should be PREFILL\n";
    return 1;
  }
  // rotary with small grid → DECODE
  if (classify_phase("rotary_embedding_kernel", GridDim{16, 1, 1}) != KernelPhase::DECODE) {
    std::cerr << "rotary small grid should be DECODE\n";
    return 1;
  }

  // Rule 4: unknown kernel → UNKNOWN
  if (classify_phase("some_generic_kernel") != KernelPhase::UNKNOWN) {
    std::cerr << "unknown kernel should be UNKNOWN\n";
    return 1;
  }

  // Custom threshold
  if (classify_phase("rms_norm_kernel", GridDim{512, 1, 1}, 256) != KernelPhase::PREFILL) {
    std::cerr << "rms_norm with custom threshold should be PREFILL\n";
    return 1;
  }

  return 0;
}
