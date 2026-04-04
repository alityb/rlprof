#pragma once

#include <cstdint>
#include <vector>

namespace hotpath {

struct PrefixGroup {
    int prefix_length = 0;
    int request_count = 0;
    double fraction = 0.0;
};

struct PrefixAnalysis {
    int total_requests = 0;
    int unique_prefixes = 0;
    double avg_requests_per_prefix = 0.0;
    int median_shared_prefix_len = 0;
    double cacheable_token_fraction = 0.0;
    std::vector<PrefixGroup> top_prefixes;
};

PrefixAnalysis analyze_prefixes(const std::vector<std::vector<int>>& prompts,
                                int min_prefix_len = 4);

}  // namespace hotpath
