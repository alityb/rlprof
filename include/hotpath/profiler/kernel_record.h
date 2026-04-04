#pragma once

#include <cstdint>
#include <string>

namespace hotpath::profiler {

struct KernelRecord {
  std::string name;
  std::string category;
  std::int64_t total_ns;
  std::int64_t calls;
  std::int64_t avg_ns;
  std::int64_t min_ns;
  std::int64_t max_ns;
  std::int64_t registers;
  std::int64_t shared_mem;
};

}  // namespace hotpath::profiler
