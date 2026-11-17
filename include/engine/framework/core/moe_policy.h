#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace engine::core {

enum class MoePlacement { Default, CpuShare, CostAware };
enum class MoeCache { Lru, FrequencyAware };

// Prepared session configuration. Each owner selects its implementation once.
struct MoePolicyConfig {
    MoePlacement placement = MoePlacement::Default;
    float cpu_share = 0;
    float cpu_cost_scale = 1;
    MoeCache cache = MoeCache::Lru;
    uint64_t decay_steps = 128;
    bool active() const noexcept {
        return hybrid() || cache != MoeCache::Lru;
    }
    bool hybrid() const noexcept {
        return placement == MoePlacement::CostAware ||
            (placement == MoePlacement::CpuShare && cpu_share > 0);
    }
};

MoePolicyConfig parse_moe_policy(const std::unordered_map<std::string, std::string> & options);

} // namespace engine::core
