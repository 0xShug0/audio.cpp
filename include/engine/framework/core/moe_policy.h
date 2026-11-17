#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

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

// Logical-layer history survives prepared graph replacement. The expert
// resource owner retains it; placement plans borrow it with their cache group.
struct MoeCpuShareState {
    uint64_t cache_epoch = 0;
    uint32_t credit = 0;
    int64_t tokens_seen = 0;
    std::vector<float> frequency;
    std::vector<int64_t> last_use;
};

MoePolicyConfig parse_moe_policy(const std::unordered_map<std::string, std::string> & options);

} // namespace engine::core
