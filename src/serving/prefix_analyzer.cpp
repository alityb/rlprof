#include "hotpath/prefix_analyzer.h"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>

namespace hotpath {
namespace {

struct TrieNode {
    std::map<int, std::unique_ptr<TrieNode>> children;
    int leaf_count = 0;  // number of prompts passing through this node
};

void insert_trie(TrieNode* root, const std::vector<int>& tokens) {
  TrieNode* node = root;
  node->leaf_count++;
  for (int token : tokens) {
    auto& child = node->children[token];
    if (!child) {
      child = std::make_unique<TrieNode>();
    }
    child->leaf_count++;
    node = child.get();
  }
}

struct SharedPrefix {
    int length;
    int count;
};

// Walk the trie to find branching points (or leaf ends) where leaf_count >= 2
void find_shared_prefixes(const TrieNode* node, int depth, int min_prefix_len,
                          std::vector<SharedPrefix>& out) {
  if (node->children.empty()) {
    // Leaf: if multiple prompts end here with sufficient length
    if (depth >= min_prefix_len && node->leaf_count >= 2) {
      out.push_back({depth, node->leaf_count});
    }
    return;
  }

  // Check if this is a branching point or a terminal point for some prompts
  bool is_branching = node->children.size() > 1;
  if (is_branching && depth >= min_prefix_len && node->leaf_count >= 2) {
    out.push_back({depth, node->leaf_count});
    return;  // Don't recurse past branching points for prefix counting
  }

  for (const auto& [token, child] : node->children) {
    find_shared_prefixes(child.get(), depth + 1, min_prefix_len, out);
  }
}

}  // namespace

PrefixAnalysis analyze_prefixes(const std::vector<std::vector<int>>& prompts,
                                int min_prefix_len) {
  PrefixAnalysis result;
  result.total_requests = static_cast<int>(prompts.size());

  if (prompts.empty()) return result;

  // Build trie
  auto root = std::make_unique<TrieNode>();
  int64_t total_prompt_tokens = 0;
  for (const auto& p : prompts) {
    insert_trie(root.get(), p);
    total_prompt_tokens += static_cast<int64_t>(p.size());
  }

  // Find shared prefixes
  std::vector<SharedPrefix> shared;
  find_shared_prefixes(root.get(), 0, min_prefix_len, shared);

  // Also count unique paths (prompts that don't share a prefix with anyone)
  // A unique prompt is one that reaches a branching point where it's the only one in its branch
  // For simplicity: unique_prefixes = number of shared prefix groups + singletons

  // Count requests covered by shared prefixes
  int shared_request_count = 0;
  int64_t cacheable_tokens = 0;
  for (const auto& sp : shared) {
    shared_request_count += sp.count;
    cacheable_tokens += static_cast<int64_t>(sp.length) * sp.count;
  }

  // Singletons: requests not part of any shared prefix
  int singletons = result.total_requests - shared_request_count;
  if (singletons < 0) singletons = 0;

  result.unique_prefixes = static_cast<int>(shared.size()) + singletons;
  if (result.unique_prefixes > 0) {
    result.avg_requests_per_prefix =
        static_cast<double>(result.total_requests) / result.unique_prefixes;
  }

  // Median shared prefix length
  if (!shared.empty()) {
    std::vector<int> lengths;
    for (const auto& sp : shared) {
      for (int i = 0; i < sp.count; ++i) {
        lengths.push_back(sp.length);
      }
    }
    std::sort(lengths.begin(), lengths.end());
    result.median_shared_prefix_len = lengths[lengths.size() / 2];
  }

  // Cacheable token fraction
  if (total_prompt_tokens > 0) {
    result.cacheable_token_fraction =
        static_cast<double>(cacheable_tokens) / total_prompt_tokens;
  }

  // Top prefixes (sorted by request count desc)
  std::sort(shared.begin(), shared.end(),
            [](const SharedPrefix& a, const SharedPrefix& b) {
              return a.count > b.count;
            });

  const size_t top_n = std::min<size_t>(10, shared.size());
  for (size_t i = 0; i < top_n; ++i) {
    result.top_prefixes.push_back(PrefixGroup{
        .prefix_length = shared[i].length,
        .request_count = shared[i].count,
        .fraction = static_cast<double>(shared[i].count) / result.total_requests,
    });
  }

  return result;
}

}  // namespace hotpath
