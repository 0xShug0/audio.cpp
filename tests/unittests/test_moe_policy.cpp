#include "engine/framework/moe_engine/moe_policy.h"
#include "engine/framework/runtime/moe_options.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
namespace moe = engine::moe;
using Options = std::unordered_map<std::string, std::string>;
void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }

void selection_contracts() {
    moe::PolicyConfig config;
    config.placement = {"placement_cpu_share", {{"share", "1"}}};
    config.eviction = {"cache_frequency_aware", {{"decay_steps", "128"}}};
    const auto prepared = moe::prepare_policy(config);
    require(prepared.requirements.cpu_execution && !prepared.requirements.phase_times, "CPU-share requirements changed");
    auto placement = prepared.placement(12);
    const std::vector<size_t> counts(12, 1);
    std::vector<int32_t> experts{0, 1, 2, 3, 4};
    const std::vector<int32_t> empty{-1, -1};
    moe::PlacementDecision decision; decision.gpu.reserve(12); decision.cpu.reserve(12);
    placement->select({experts, counts, empty, 0, 4096, 1}, decision);
    require(decision.gpu == std::vector<int32_t>({0, 1}) && decision.cpu == std::vector<int32_t>({2, 3, 4}), "CPU-share cold fills differ from independent expected placement");
    experts = {3, 1, 4, 0, 2};
    const std::vector<int32_t> resident{0, 1};
    placement->select({experts, counts, resident, 0, 4096, 1}, decision);
    require(decision.gpu == std::vector<int32_t>({1, 0}) && decision.cpu == std::vector<int32_t>({3, 4, 2}), "CPU-share moved a fixed GPU hit or changed miss order");
    config.placement.parameters["share"] = "0.5";
    placement = moe::prepare_policy(config).placement(12);
    experts = {2, 3, 4};
    placement->select({experts, counts, resident, 0, 4096, 1}, decision);
    require(decision.gpu == std::vector<int32_t>({2, 4}) && decision.cpu == std::vector<int32_t>({3}), "CPU-share fractional carry differs from scalar expectation");
    experts = {5}; placement->select({experts, counts, resident, 0, 4096, 2}, decision);
    require(decision.cpu == experts, "CPU-share carry was lost across prepared programs");
    placement->event({moe::PolicyEvent::CacheInvalidated, 0, 1});
    placement->select({experts, counts, resident, 1, 4096, 2}, decision);
    require(decision.gpu == experts && decision.cpu.empty(), "cache invalidation did not reset CPU-share carry");

    auto eviction = prepared.eviction(12);
    eviction->accessed({2, 10}, {0, 0, resident}); eviction->accessed({3, 1}, {0, 0, resident});
    require(eviction->select({{{0, 2, 1}, {1, 3, 2}}, {0, 0, resident}}) == 1, "frequency eviction ignored observed assignment demand");
    require(eviction->select({{{0, 2, 1}}, {0, 0, resident}}) == 0, "frequency eviction selected a protected/non-candidate slot");
    config.placement = {"placement_cost_aware", {{"cpu_ms_per_row", "0.01"}, {"gpu_ms_per_row", "0.001"}, {"upload_bytes_per_ms", "1000"}}};
    const auto cost = moe::prepare_policy(config);
    require(cost.requirements.cpu_execution && cost.requirements.phase_times && bool(cost.eviction), "independent policy preparation lost combined requirements");
    auto policy = cost.placement(12); policy->prepare_program(7);
    experts = {0, 2};
    policy->select({experts, counts, resident, 0, 4096, 7}, decision);
    require(decision.gpu == std::vector<int32_t>({0}) && decision.cpu == std::vector<int32_t>({2}), "cost policy differs from explicit transfer/compute cost comparison");
    policy->event({moe::PolicyEvent::ProgramRetired, 7});
    policy->prepare_program(8);
    policy->select({experts, counts, resident, 0, 4096, 8}, decision);
    require(decision.cpu == std::vector<int32_t>({2}), "cost program retirement lost configured priors");
}
void option_contracts() {
    using engine::runtime::resolve_moe_options;
    for (const auto & family : {"fish_drama3", "ming3_audio", "nemotron_transcribe", "zonos2", "zonos3"}) {
        moe::EngineConfig defaults;
        const auto baseline = resolve_moe_options({}, family, defaults, 6);
        require(baseline.residency == defaults.residency && baseline.cache_slots == defaults.cache_slots && baseline.cpu_layers == defaults.cpu_layers, "omitted controls changed model defaults");
        const std::string prefix = std::string(family) + ".";
        const auto translated = resolve_moe_options({{"moe_disk_experts", "true"}, {prefix + "pin_experts", "false"},
            {"moe_expert_cache_size", "4"}, {prefix + "disk_expert_read_threads", "2"},
            {"moe_cpu_expert_layers", "2"}, {"moe_cpu_share", "0.5"}}, family, defaults, 6);
        require(translated.residency == moe::Residency::Staged && !translated.pin_host_weights && translated.cache_slots == 4 && translated.read_threads == 2 && translated.cpu_layers == std::vector<size_t>({0, 1}), "legacy controls lost residency/whole-layer semantics");
        require(translated.policies.placement.name == "placement_cpu_share" && translated.policies.placement.parameters.at("share") == "0.5", "legacy CPU share was not translated to placement selection");
        bool conflict = false;
        try { (void)resolve_moe_options({{"moe_disk_experts", "true"}, {prefix + "disk_experts", "false"}}, family, defaults, 6); }
        catch (const std::invalid_argument &) { conflict = true; }
        require(conflict, "shared/family option conflict was silently accepted");
    }
    const auto combined = engine::runtime::resolve_moe_policies({{"moe_policy", "placement_cpu_share,cache_frequency_aware"},
        {"moe_policy_params.placement.share", "0.5"}, {"moe_policy_params.eviction.decay_steps", "128"}});
    require(combined.placement.parameters.at("share") == "0.5" && combined.eviction.parameters.at("decay_steps") == "128", "stage-scoped parameters collided");
    const std::vector<Options> invalid{
        {{"moe_cpu_expert_layers", "-1"}}, {{"test.cpu_expert_layers", "5"}},
        {{"moe_expert_cache_size", "0"}}, {{"moe_disk_expert_read_threads", "65"}},
        {{"moe_cpu_share", "-0.1"}}, {{"moe_cpu_share", "1.1"}}, {{"moe_cpu_share", "nan"}},
        {{"moe_policy", "placement_cpu_share,placement_cost_aware"}}, {{"moe_policy", "cache_frequency_aware,cache_frequency_aware"}},
        {{"moe_policy", "unknown"}}, {{"moe_policy_params.share", "0.5"}},
        {{"moe_policy_params.placement.typo", "0.5"}}, {{"moe_policy_params.eviction.decay_steps", "0"}}};
    for (const auto & options : invalid) {
        bool rejected = false;
        try {
            const auto resolved = resolve_moe_options(options, "test", {}, 4);
            (void)moe::prepare_policy(resolved.policies);
        } catch (const std::exception &) { rejected = true; }
        require(rejected, "invalid policy or parameter was accepted");
    }
    for (const auto * zero : {"0", "0.0", "-0", "0e0"}) {
        const auto resolved = engine::runtime::resolve_moe_policies({{"moe_cpu_share", zero}});
        require(!moe::prepare_policy(resolved).active_placement, "explicit zero became an active placement policy");
    }
}
}
int main() {
    const auto start = std::chrono::steady_clock::now();
    try {
        engine::debug::configure_logging({true, std::nullopt});
        selection_contracts(); option_contracts();
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        std::cout << "moe_policy: PASS\n"; return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
