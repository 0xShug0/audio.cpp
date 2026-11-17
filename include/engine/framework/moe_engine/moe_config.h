#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::core {

enum class MoeExpertResidency { Backend, Staged };

struct MoeExecutionPolicy {
    MoeExpertResidency residency = MoeExpertResidency::Backend;
    bool pin_host_weights = false;
    int64_t cache_slots = 8;
    int64_t read_threads = 1;
    // MoE ordinals, not backbone indices. Registration supplies their mapping.
    std::vector<size_t> cpu_layers;
};

// Resolve shared session controls and existing family-prefixed aliases using
// model-provided defaults. Supplying both spellings is an explicit error.
MoeExecutionPolicy resolve_moe_execution_policy(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view family, MoeExecutionPolicy defaults, size_t moe_layers);

} // namespace engine::core
