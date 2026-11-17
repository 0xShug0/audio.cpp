#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include "engine/framework/moe_engine/cache_policy.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::modules {
class MoePlacementPolicy;
struct StagedMoeConfig;
struct MoeExpertConfig;
using MoeLayerPlacementPolicy = std::function<std::shared_ptr<MoePlacementPolicy>(const StagedMoeConfig &, const MoeExpertConfig &)>;
}

namespace engine::core {

struct MoePlacementRequirements {
    bool cpu = false;
    bool timing = false;
};

using MoePlacementFactory = std::function<modules::MoeLayerPlacementPolicy(size_t experts)>;
struct MoePlacementDescriptor {
    MoePlacementRequirements requirements;
    std::function<MoePlacementFactory()> create;
};

// Prepared session configuration. Each owner selects its implementation once.
struct MoePolicyConfig {
    MoePlacementDescriptor placement;
    // Empty selects the store's existing LRU default.
    MoeCacheFactory cache;
    MoePlacementRequirements placement_requirements() const noexcept { return placement.requirements; }
    bool active() const noexcept {
        return has_placement() || static_cast<bool>(cache);
    }
    bool has_placement() const noexcept { return static_cast<bool>(placement.create); }
};

MoePolicyConfig parse_moe_policy(const std::unordered_map<std::string, std::string> & options);

} // namespace engine::core
