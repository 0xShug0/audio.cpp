#include "engine/framework/runtime/moe_options.h"
#include "engine/framework/runtime/options.h"
#include <algorithm>
#include <stdexcept>

namespace engine::runtime {
moe::PolicyConfig resolve_moe_policies(const std::unordered_map<std::string, std::string> & options) {
    moe::PolicyConfig config;
    if (const auto selected = find_option(options, {"moe_policy"})) {
        bool placement = false, eviction = false;
        size_t offset = 0;
        while (offset <= selected->size()) {
            const auto end = selected->find(',', offset);
            const auto name = selected->substr(offset, end == std::string::npos ? end : end - offset);
            if (name == "default") {
                if (offset || end != std::string::npos) throw std::invalid_argument("default MoE policy cannot be combined with another selection");
            } else if (name.rfind("placement_", 0) == 0) {
                if (placement) throw std::invalid_argument("multiple MoE placement policies selected");
                config.placement.name = name; placement = true;
            } else if (name.rfind("cache_", 0) == 0) {
                if (eviction) throw std::invalid_argument("multiple MoE eviction policies selected");
                config.eviction.name = name; eviction = true;
            } else throw std::invalid_argument("unknown MoE policy: " + name);
            if (end == std::string::npos) break;
            offset = end + 1;
        }
    }
    for (const auto & item : options) {
        constexpr std::string_view placement = "moe_policy_params.placement.";
        constexpr std::string_view eviction = "moe_policy_params.eviction.";
        if (item.first.rfind(placement, 0) == 0) config.placement.parameters.emplace(item.first.substr(placement.size()), item.second);
        else if (item.first.rfind(eviction, 0) == 0) config.eviction.parameters.emplace(item.first.substr(eviction.size()), item.second);
        else if (item.first.rfind("moe_policy_params.", 0) == 0)
            throw std::invalid_argument("MoE policy parameters require placement or eviction scope: " + item.first);
    }
    if (const auto share = find_option(options, {"moe_cpu_share"})) {
        if (config.placement.name != "default" || !config.placement.parameters.empty())
            throw std::invalid_argument("moe_cpu_share conflicts with an explicit placement policy");
        config.placement.name = "placement_cpu_share";
        config.placement.parameters.emplace("share", *share);
    }
    return config;
}
moe::EngineConfig resolve_moe_options(
    const std::unordered_map<std::string, std::string> & options,
    std::string_view family, moe::EngineConfig policy, size_t moe_layers) {
    const std::string prefix = std::string(family) + ".";
    const auto alias = [&](std::string_view shared, std::string_view local) {
        const std::string legacy = prefix + std::string(local);
        if (options.count(std::string(shared)) && options.count(legacy))
            throw std::invalid_argument("MoE options contain both " + std::string(shared) + " and " + legacy);
        return legacy;
    };
    const auto disk = alias("moe_disk_experts", "disk_experts");
    if (const auto value = find_option(options, {"moe_disk_experts", disk}))
        policy.residency = parse_bool_option(*value, "moe_disk_experts")
            ? moe::Residency::Staged : moe::Residency::Resident;
    const auto pin = alias("moe_pin_experts", "pin_experts");
    if (const auto value = find_option(options, {"moe_pin_experts", pin}))
        policy.pin_host_weights = parse_bool_option(*value, "moe_pin_experts");
    const auto cache = alias("moe_expert_cache_size", "expert_cache_size");
    policy.cache_slots = parse_positive_i64_option(options, {"moe_expert_cache_size", cache}, policy.cache_slots);
    const auto readers = alias("moe_disk_expert_read_threads", "disk_expert_read_threads");
    policy.read_threads = parse_positive_i64_option(options,
        {"moe_disk_expert_read_threads", readers}, policy.read_threads);
    if (policy.read_threads > 64) throw std::invalid_argument("MoE disk expert read threads must be between 1 and 64");
    const auto cpu = alias("moe_cpu_expert_layers", "cpu_expert_layers");
    if (const auto value = parse_i64_option(options, {"moe_cpu_expert_layers", cpu})) {
        if (*value < 0 || static_cast<uint64_t>(*value) > moe_layers)
            throw std::invalid_argument("MoE CPU expert layer count exceeds the number of MoE layers");
        policy.cpu_layers.resize(static_cast<size_t>(*value));
        for (size_t i = 0; i < policy.cpu_layers.size(); ++i) policy.cpu_layers[i] = i;
    }
    const auto explicit_policies = resolve_moe_policies(options);
    if (options.count("moe_policy") || options.count("moe_cpu_share") || std::any_of(options.begin(), options.end(), [](const auto & item) {
        return item.first.rfind("moe_policy_params.", 0) == 0;
    })) policy.policies = explicit_policies;
    auto layers = policy.cpu_layers;
    std::sort(layers.begin(), layers.end());
    if (std::adjacent_find(layers.begin(), layers.end()) != layers.end() ||
        (!layers.empty() && layers.back() >= moe_layers))
        throw std::invalid_argument("invalid MoE CPU layer ordinals");
    return policy;
}

} // namespace engine::runtime
