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

void selection_contracts(bool existing_only) {
    moe::PolicyConfig config;
    config.placement = {"placement_cpu_share", {{"share", "1"}}};
    config.eviction = {"cache_frequency_aware", {{"decay_steps", "128"}}};
    const auto prepared = moe::prepare_policy(config);
    require(prepared.requirements.cpu_execution && !prepared.requirements.phase_times, "CPU-share requirements changed");
    auto placement = prepared.placement(12);
    placement->prepare_program(1);
    const std::vector<size_t> counts(12, 1);
    std::vector<int32_t> experts{0, 1, 2, 3, 4};
    const std::vector<int32_t> empty{-1, -1};
    moe::PlacementDecision decision; decision.gpu.reserve(12); decision.cpu.reserve(12);
    placement->select({experts, counts, empty, 0, 4096, 1}, decision);
    require(decision.gpu == std::vector<int32_t>({0, 1}) && decision.cpu == std::vector<int32_t>({2, 3, 4}), "CPU-share cold fills differ from independent expected placement");
    experts = {3, 1, 4, 0, 2};
    const std::vector<int32_t> resident{0, 1};
    placement->select({experts, counts, resident, 0, 4096, 1}, decision);
    require(decision.gpu == std::vector<int32_t>({0, 1}) && decision.cpu == std::vector<int32_t>({2, 3, 4}), "CPU-share did not retain legacy count/ID order and fixed GPU hits");
    config.placement.parameters["share"] = "0.5";
    placement = moe::prepare_policy(config).placement(12);
    placement->prepare_program(1); placement->prepare_program(2);
    experts = {2, 3, 4};
    placement->select({experts, counts, resident, 0, 4096, 1}, decision);
    require(decision.gpu == std::vector<int32_t>({2, 4}) && decision.cpu == std::vector<int32_t>({3}), "CPU-share fractional carry differs from scalar expectation");
    experts = {5}; placement->select({experts, counts, resident, 0, 4096, 2}, decision);
    require(decision.gpu == experts && decision.cpu.empty(), "CPU-share leaked prefill carry into a different program");
    placement->select({experts, counts, resident, 0, 4096, 1}, decision);
    require(decision.cpu == experts, "CPU-share lost carry when returning to the original program");
    placement->event({moe::PolicyEvent::CacheInvalidated, 0, 1});
    placement->select({experts, counts, resident, 1, 4096, 2}, decision);
    require(decision.gpu == experts && decision.cpu.empty(), "cache invalidation did not reset CPU-share carry");

    placement = moe::prepare_policy(config).placement(12);
    placement->prepare_program(3);
    experts = {0, 2, 3, 4};
    const std::vector<int32_t> partly_resident{0, -1};
    placement->select({experts, counts, partly_resident, 0, 4096, 3}, decision);
    require(decision.gpu == std::vector<int32_t>({0, 2, 3}) && decision.cpu == std::vector<int32_t>({4}),
        "CPU-share did not preserve a hit, fill its vacant slot, then split remaining misses");

    if (existing_only) return;

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
void cpu_share_ranking_and_lifecycle() {
    moe::PolicyConfig config;
    config.placement = {"placement_cpu_share", {{"share", "0.5"}}};
    const auto prepared = moe::prepare_policy(config);
    auto policy = prepared.placement(10);
    const std::vector<size_t> counts{1, 65, 1, 33, 1, 17, 1, 9, 0, 0};
    const std::vector<int32_t> resident{8, 9};
    const std::vector<std::vector<int32_t>> orders{{0, 1, 2, 3, 4, 5, 6, 7}, {7, 4, 2, 6, 3, 0, 5, 1}};
    moe::PlacementDecision decision;
    decision.cpu.reserve(10); decision.gpu.reserve(10);
    for (size_t i = 0; i < orders.size(); ++i) {
        policy->prepare_program(i);
        policy->select({orders[i], counts, resident, 0, 4096, i}, decision);
        require(decision.cpu == std::vector<int32_t>({3, 7, 2, 6}) &&
            decision.gpu == std::vector<int32_t>({1, 5, 0, 4}),
            "CPU-share depends on route occurrence instead of legacy assignment-count ranking");
        size_t rows = 0; for (auto expert : decision.cpu) rows += counts[expert];
        require(rows == 44, "skewed placement selected the heavy 124-row CPU workload");
    }
    const std::vector<int32_t> miss{5};
    const std::vector<size_t> single_counts(10, 1);
    policy->prepare_program(10); policy->prepare_program(11);
    policy->select({miss, single_counts, resident, 0, 4096, 10}, decision);
    require(decision.cpu.empty(), "first miss should remain on GPU");
    policy->select({miss, single_counts, resident, 0, 4096, 11}, decision);
    require(decision.cpu.empty(), "independent program inherited fractional carry");
    policy->select({miss, single_counts, resident, 0, 4096, 10}, decision);
    require(decision.cpu == miss, "returning program did not retain its own carry");
    policy->event({moe::PolicyEvent::ProgramRetired, 11});
    policy->prepare_program(11);
    policy->select({miss, single_counts, resident, 0, 4096, 11}, decision);
    require(decision.cpu.empty(), "retired program retained fractional carry after reprepare");
    policy->event({moe::PolicyEvent::CacheInvalidated, 0, 1});
    policy->select({miss, single_counts, resident, 1, 4096, 11}, decision);
    require(decision.cpu.empty(), "cache invalidation retained carry");
    policy->select({miss, single_counts, resident, 2, 4096, 11}, decision);
    require(decision.cpu.empty(), "new cache epoch retained carry without a notification");
    policy->event({moe::PolicyEvent::ProgramRetired, 11});
    bool rejected = false;
    try { policy->select({miss, single_counts, resident, 2, 4096, 11}, decision); }
    catch (const std::out_of_range &) { rejected = true; }
    require(rejected, "retired program silently recreated execution state");
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
int main(int argc, char ** argv) {
    const auto start = std::chrono::steady_clock::now();
    try {
        engine::debug::configure_logging({true, std::nullopt});
        const bool existing_only = argc > 1 && std::string(argv[1]) == "--existing-policies-only";
        cpu_share_ranking_and_lifecycle(); selection_contracts(existing_only); option_contracts();
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        std::cout << "moe_policy: PASS\n"; return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
