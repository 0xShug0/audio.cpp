#include "engine/framework/moe_engine/moe_policy.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/moe_engine/cache_policy.h"
#include "placement_policy_internal.h"

#include <stdexcept>
#include <string_view>

namespace engine::core {
namespace {
using Parameters = std::unordered_map<std::string, std::string>;

MoePolicyConfig placement_cpu_share(const Parameters & params) {
    for (const auto & item : params)
        if (item.first != "share") throw std::invalid_argument("unknown placement_cpu_share parameter: " + item.first);
    const auto share = runtime::parse_finite_float_option(params, {"share"});
    if (!share)
        throw std::invalid_argument("placement_cpu_share requires moe_policy_params.share between zero and one");
    MoePolicyConfig result;
    result.placement = modules::prepare_cpu_share(*share);
    return result;
}

MoePolicyConfig placement_cost_aware(const Parameters & params) {
    for (const auto & item : params)
        if (item.first != "cpu_cost_scale") throw std::invalid_argument("unknown placement_cost_aware parameter: " + item.first);
    MoePolicyConfig result;
    result.placement = modules::prepare_cost_aware(runtime::parse_finite_float_option(params, {"cpu_cost_scale"}).value_or(1));
    return result;
}

MoePolicyConfig cache_frequency_aware(const Parameters & params) {
    for (const auto & item : params)
        if (item.first != "decay_steps") throw std::invalid_argument("unknown cache_frequency_aware parameter: " + item.first);
    MoePolicyConfig result;
    const auto decay = static_cast<uint64_t>(runtime::parse_positive_i64_option(params, {"decay_steps"}, 128));
    result.cache = prepare_moe_frequency_cache(decay);
    return result;
}
} // namespace

MoePolicyConfig parse_moe_policy(const std::unordered_map<std::string, std::string> & options) {
    if (options.count("moe_cpu_share"))
        throw std::invalid_argument("moe_cpu_share was removed; use moe_policy=placement_cpu_share and moe_policy_params.share");
    Parameters params;
    constexpr std::string_view prefix = "moe_policy_params.";
    for (const auto & item : options)
        if (item.first.compare(0, prefix.size(), prefix) == 0)
            params.emplace(item.first.substr(prefix.size()), item.second);
    const auto name = options.find("moe_policy");
    if (name == options.end() || name->second == "default") {
        if (!params.empty()) throw std::invalid_argument("moe_policy_params requires an active moe_policy");
        return {};
    }
    // Public policy registration. Algorithm dispatch stays in its owning component.
    static const struct { const char * name; MoePolicyConfig (*prepare)(const Parameters &); } policies[] = {
        {"placement_cpu_share", placement_cpu_share},
        {"placement_cost_aware", placement_cost_aware},
        {"cache_frequency_aware", cache_frequency_aware},
    };
    for (const auto & policy : policies) {
        if (name->second != policy.name) continue;
        auto result = policy.prepare(params);
        return result;
    }
    throw std::invalid_argument("unknown moe_policy: " + name->second);
}
} // namespace engine::core
