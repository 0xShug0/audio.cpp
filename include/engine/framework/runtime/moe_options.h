#pragma once
#include "engine/framework/moe_engine/moe_engine.h"
#include <string_view>
#include <unordered_map>

namespace engine::runtime {
moe::EngineConfig resolve_moe_options(const std::unordered_map<std::string, std::string> & options,
    std::string_view family, moe::EngineConfig defaults, size_t moe_layers);
// Pure syntax/stage resolution for application-side option forwarding checks.
moe::PolicyConfig resolve_moe_policies(const std::unordered_map<std::string, std::string> & options);
} // namespace engine::runtime
