#include "engine/framework/moe_engine/moe_config.h"
#include "engine/framework/moe_engine/moe_policy.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::core {

MoeExecutionPolicy resolve_moe_execution_policy(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view family, MoeExecutionPolicy policy, size_t moe_layers) {
    const std::string prefix = std::string(family) + ".";
    const auto alias = [&](std::string_view shared, std::string_view local) {
        const std::string legacy = prefix + std::string(local);
        if (options.count(std::string(shared)) && options.count(legacy))
            throw std::invalid_argument("MoE options contain both " + std::string(shared) + " and " + legacy);
        return legacy;
    };
    const auto disk = alias("moe_disk_experts", "disk_experts");
    if (const auto value = runtime::find_option(options, {"moe_disk_experts", disk}))
        policy.residency = runtime::parse_bool_option(*value, "moe_disk_experts")
            ? MoeExpertResidency::Staged : MoeExpertResidency::Backend;
    const auto pin = alias("moe_pin_experts", "pin_experts");
    if (const auto value = runtime::find_option(options, {"moe_pin_experts", pin}))
        policy.pin_host_weights = runtime::parse_bool_option(*value, "moe_pin_experts");
    const auto cache = alias("moe_expert_cache_size", "expert_cache_size");
    policy.cache_slots = runtime::parse_positive_i64_option(options, {"moe_expert_cache_size", cache}, policy.cache_slots);
    const auto readers = alias("moe_disk_expert_read_threads", "disk_expert_read_threads");
    policy.read_threads = runtime::parse_positive_i64_option(options,
        {"moe_disk_expert_read_threads", readers}, policy.read_threads);
    if (policy.read_threads > 64) throw std::invalid_argument("MoE disk expert read threads must be between 1 and 64");
    const auto cpu = alias("moe_cpu_expert_layers", "cpu_expert_layers");
    if (const auto value = runtime::parse_i64_option(options, {"moe_cpu_expert_layers", cpu})) {
        if (*value < 0 || static_cast<uint64_t>(*value) > moe_layers)
            throw std::invalid_argument("MoE CPU expert layer count exceeds the number of MoE layers");
        policy.cpu_layers.resize(static_cast<size_t>(*value));
        for (size_t i = 0; i < policy.cpu_layers.size(); ++i) policy.cpu_layers[i] = i;
    }
    const auto selected_policy = parse_moe_policy(options);
    if (selected_policy.active() && policy.residency != MoeExpertResidency::Staged)
        throw std::invalid_argument("selected moe_policy requires staged MoE experts");
    auto layers = policy.cpu_layers;
    std::sort(layers.begin(), layers.end());
    if (std::adjacent_find(layers.begin(), layers.end()) != layers.end() ||
        (!layers.empty() && layers.back() >= moe_layers))
        throw std::invalid_argument("invalid MoE CPU layer ordinals");
    return policy;
}

} // namespace engine::core
