#include "engine/framework/moe_engine/moe_policy.h"

#include <stdexcept>

namespace engine::moe {
PreparedPolicy prepare_cpu_share(const PolicyParameters &);
PreparedPolicy prepare_cost_aware(const PolicyParameters &);
PreparedPolicy prepare_lru(const PolicyParameters &);
PreparedPolicy prepare_frequency(const PolicyParameters &);

PreparedPolicy prepare_policy(const PolicyConfig & config) {
    using PlacementFactory = PreparedPolicy (*)(const PolicyParameters &);
    static const std::unordered_map<std::string, PlacementFactory> placements{
        {"default", prepare_cpu_share}, {"placement_cpu_share", prepare_cpu_share},
        {"placement_cost_aware", prepare_cost_aware}};
    using CacheFactory = PreparedPolicy (*)(const PolicyParameters &);
    static const std::unordered_map<std::string, CacheFactory> evictions{
        {"default", prepare_lru}, {"cache_frequency_aware", prepare_frequency}};
    const auto placement = placements.find(config.placement.name);
    if (placement == placements.end()) throw std::invalid_argument("unknown MoE placement policy: " + config.placement.name);
    const auto eviction = evictions.find(config.eviction.name);
    if (eviction == evictions.end()) throw std::invalid_argument("unknown MoE eviction policy: " + config.eviction.name);
    auto prepared = placement->second(config.placement.parameters);
    auto cache = eviction->second(config.eviction.parameters);
    prepared.requirements.cpu_execution |= cache.requirements.cpu_execution;
    prepared.requirements.phase_times |= cache.requirements.phase_times;
    prepared.requirements.cache_accesses |= cache.requirements.cache_accesses;
    prepared.requirements.completed_work |= cache.requirements.completed_work || prepared.requirements.phase_times;
    prepared.eviction = std::move(cache.eviction);
    return prepared;
}
} // namespace engine::moe
