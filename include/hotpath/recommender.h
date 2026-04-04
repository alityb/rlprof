#pragma once

#include <string>

#include "hotpath/disagg_model.h"

namespace hotpath {

std::string generate_vllm_config(const DisaggEstimate& estimate, const std::string& model);
std::string generate_llmd_config(const DisaggEstimate& estimate, const std::string& model);
std::string generate_dynamo_config(const DisaggEstimate& estimate, const std::string& model);
std::string generate_summary(const DisaggEstimate& estimate, const std::string& model);

}  // namespace hotpath
