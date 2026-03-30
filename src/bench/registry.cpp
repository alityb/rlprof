#include "rlprof/bench/registry.h"

#include <unordered_map>

#include "rlprof/bench/kernels/rms_norm.h"
#include "rlprof/bench/kernels/rotary_emb.h"
#include "rlprof/bench/kernels/silu_mul.h"

namespace rlprof::bench {
namespace {

std::unordered_map<std::string, std::vector<KernelImpl>>& registry() {
  static std::unordered_map<std::string, std::vector<KernelImpl>> value;
  return value;
}

}  // namespace

void register_kernel(const std::string& category, const KernelImpl& implementation) {
  registry()[category].push_back(implementation);
}

std::vector<KernelImpl> get_kernel_impls(const std::string& category) {
  if (!registry().contains(category)) {
    return {};
  }
  return registry().at(category);
}

void clear_registry() {
  registry().clear();
}

void register_builtin_kernels() {
  if (!registry().empty()) {
    return;
  }
  kernels::register_silu_mul();
  kernels::register_rms_norm();
  kernels::register_rotary_emb();
}

}  // namespace rlprof::bench
