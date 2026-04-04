#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace hotpath::bench {

using Shape = std::vector<std::int64_t>;

struct KernelImpl {
  std::string name;
  std::function<std::any(const Shape&, const std::string&)> setup;
  std::function<void(std::any&)> fn;
  std::vector<std::string> dtypes;
  std::function<std::size_t(const Shape&, const std::string&)> bytes_processed;
};

void register_kernel(const std::string& category, const KernelImpl& implementation);

std::vector<KernelImpl> get_kernel_impls(const std::string& category);

void clear_registry();

void register_builtin_kernels();

}  // namespace hotpath::bench
