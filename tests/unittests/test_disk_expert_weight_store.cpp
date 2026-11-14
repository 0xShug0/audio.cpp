#include "test_allocation_failure.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/moe_resources.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/modules/moe_module.h"
#include "engine/framework/modules/staged_moe_runtime.h"
#include <ggml-alloc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace {
struct FixtureDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("audiocpp_disk_experts_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FixtureDirectory() { std::filesystem::create_directories(path); }
    ~FixtureDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
struct ContextDeleter { void operator()(ggml_context * ctx) const { ggml_free(ctx); } };
struct BufferDeleter { void operator()(ggml_backend_buffer_t buffer) const { ggml_backend_buffer_free(buffer); } };
struct SchedulerDeleter { void operator()(ggml_backend_sched_t scheduler) const { ggml_backend_sched_free(scheduler); } };

void check_group_registration_failures(const std::filesystem::path & path,
    engine::core::ExecutionContext & execution, const std::vector<std::string> & names,
    bool pinned, engine::core::ExecutionContext * cpu_execution) {
    namespace core = engine::core;
    size_t failures = 0;
    bool completed = false;
    for (size_t allocation = 0; allocation < 64; ++allocation) {
        {
            core::DiskExpertWeightStore store(path, pinned, 1, cpu_execution);
            const auto first = store.add_group(names, execution.backend(), 2);
            const auto slots = store.stage(first, {0, 1});
            const auto before = store.statistics();
            std::unique_ptr<ggml_context, ContextDeleter> context(
                ggml_init({ggml_tensor_overhead() + 1024, nullptr, true}));
            if (!context) throw std::runtime_error("cannot create failed-registration route fixture");
            auto * unpublished = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 1);
            ggml_set_name(unpublished, "disk_experts.route.1");
            size_t second = 0;
            bool rejected = false;
            try {
                engine::test::AllocationFailure fault(allocation);
                second = store.add_group(names, execution.backend(), 4);
                fault.disable();
                completed = true;
            } catch (const std::bad_alloc &) {
                rejected = true;
                ++failures;
            }
            if (rejected) {
                const auto after = store.statistics();
                if (!engine::test::AllocationFailure::triggered() ||
                    core::DiskExpertWeightStore::evaluate(unpublished, true, &store) ||
                    std::tie(after.bytes_read, after.expert_loads, after.cache_hits, after.buffer_bytes,
                        after.host_buffer_bytes, after.read_ms, after.upload_ms, after.stage_ms) !=
                    std::tie(before.bytes_read, before.expert_loads, before.cache_hits, before.buffer_bytes,
                        before.host_buffer_bytes, before.read_ms, before.upload_ms, before.stage_ms))
                    throw std::runtime_error("failed expert registration changed committed statistics");
                // Force a miss through the old staging buffer after failure,
                // rather than checking only cache hits before the retry.
                store.stage(first, {2, 3});
                second = store.add_group(names, execution.backend(), 4);
            }
            if (first != 0 || second != 1 || store.stage(first, {0, 1}) != slots ||
                store.stage(second, {8, 9, 10, 11}).size() != 4)
                throw std::runtime_error("expert registration retry lost groups, cache slots or staging");
        }
        if (engine::test::AllocationFailure::outstanding() != 0)
            throw std::runtime_error("expert registration failure/retry leaked an allocation at fault " + std::to_string(allocation));
        if (completed) break;
    }
    if (!completed || failures == 0)
        throw std::runtime_error("expert registration fixture did not reach failure and recovery");
    std::cout << "group_registration_failures pinned=" << pinned << " failures=" << failures << '\n';
}

void check_host_bank_failures(const std::filesystem::path & path, engine::core::ExecutionContext & execution,
    const std::vector<std::string> & names, const std::vector<engine::core::TensorValue> & reference, bool pinned,
    engine::core::ExecutionContext * cpu_execution) {
    namespace core = engine::core;
    std::vector<std::vector<std::byte>> expected;
    for (const auto & value : reference) {
        expected.emplace_back(ggml_nbytes(value.tensor));
        ggml_backend_tensor_get(value.tensor, expected.back().data(), 0, expected.back().size());
    }
    size_t failures = 0;
    bool completed = false;
    for (size_t allocation = 0; allocation < 32; ++allocation) {
        {
            core::DiskExpertWeightStore store(path, pinned, 1, cpu_execution);
            const auto group = store.add_group(names, execution.backend(), 4);
            bool rejected = false;
            try {
                engine::test::AllocationFailure fault(allocation);
                store.host_tensor(group, store.tensor(group, names.size() - 1).tensor);
                fault.disable();
                completed = true;
            } catch (const std::bad_alloc &) {
                rejected = true;
                ++failures;
            }
            if (rejected && !engine::test::AllocationFailure::triggered())
                throw std::runtime_error("unexpected host-bank allocation failure");
            // Retry every projection, including those not reached by a failed
            // initialization. Native bytes and stable bindings must survive.
            for (size_t projection = 0; projection < names.size(); ++projection) {
                auto * bank = store.tensor(group, projection).tensor;
                const auto recovered = store.host_tensor(group, bank);
                if (recovered.tensor->type != reference[projection].tensor->type ||
                    ggml_nbytes(recovered.tensor) != expected[projection].size())
                    throw std::runtime_error("host-bank retry changed native projection layout");
                std::vector<std::byte> bytes(expected[projection].size());
                ggml_backend_tensor_get(recovered.tensor, bytes.data(), 0, bytes.size());
                if (bytes != expected[projection] || store.host_tensor(group, bank).tensor != recovered.tensor)
                    throw std::runtime_error("host-bank retry changed weights or rebuilt a completed bank");
            }
        }
        if (engine::test::AllocationFailure::outstanding() != 0)
            throw std::runtime_error("host-bank failure/retry leaked a buffer wrapper or allocation");
        if (completed) break;
    }
    if (!completed || failures < names.size() + 2)
        throw std::runtime_error("host-bank failure fixture missed projection allocation boundaries");
    std::cout << "host_bank_allocation_failures type=" << ggml_type_name(reference[0].tensor->type)
              << " pinned=" << pinned << " failures=" << failures << '\n';
}

void automatic_hybrid_graph(engine::core::ExecutionContext & execution, engine::core::ExecutionContext & host,
    engine::core::DiskExpertWeightStore & store, size_t group, const std::vector<engine::core::TensorValue> & weights) {
    namespace core = engine::core;
    namespace modules = engine::modules;
    const bool quantized = weights[0].tensor->type != GGML_TYPE_F32;
    for (bool relu : {false, true}) for (bool rounded : {false, true}) {
        modules::MoeExpertConfig math;
        math.fused_swiglu = !relu && !rounded;
        math.activation = relu ? modules::MoeActivation::ReluSquared : modules::MoeActivation::SwiGlu;
        math.round_projections = math.round_activation = math.round_product = math.round_output = rounded;
        modules::MoeModule module(math);
        core::PreparedGraph owner(execution, {"test.moe.auto", 4 * 1024 * 1024, 256, 16 * ggml_tensor_overhead()});
        auto & persistent = owner.persistent_build_context();
        auto hidden = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({3, 256}));
        auto logical = core::make_tensor(persistent, GGML_TYPE_I32, core::TensorShape::from_dims({3, 2}));
        auto mixing = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({3, 2}));
        auto counter = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        auto one = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        for (auto tensor : {hidden, logical, mixing, counter, one}) ggml_set_input(tensor.tensor);
        owner.allocate_persistent();
        // Copy the explicit recorder when composing a module build context.
        auto ctx = owner.build_context();
        ctx.module_instance_name = "manual_moe";
        auto * bank_ids = store.route(ctx, logical.tensor, group);
        auto * advance = ggml_cpy(ctx.ggml, ggml_add(ctx.ggml, counter.tensor, one.tensor), counter.tensor);
        auto * prefix = ggml_add(ctx.ggml, hidden.tensor, ggml_scale(ctx.ggml, advance, 0.0F));
        auto * output = module.build(ctx, prefix, {logical.tensor, mixing.tensor},
            {store.tensor(group, 1).tensor, relu ? nullptr : store.tensor(group, 0).tensor, store.tensor(group, 2).tensor}, bank_ids);
        ggml_set_output(output);
        ggml_build_forward_expand(owner.graph(), output);
        auto * scheduler = owner.create_scheduler({execution.backend(), host.backend()});
        ggml_backend_sched_set_eval_callback(scheduler, core::DiskExpertWeightStore::evaluate, &store);
        if (!core::DiskExpertWeightStore::evaluate(bank_ids, true, &store))
            throw std::runtime_error("automatic MoE changed placement discovery before preparation");
        owner.allocate();
        core::PreparedGraph reference(execution, {"test.moe.auto.reference", 4 * 1024 * 1024, 256});
        auto & ref = reference.build_context();
        auto * gpu_lanes = module.build_experts(ref,
            ggml_reshape_3d(ref.ggml, hidden.tensor, 256, 1, 3), logical.tensor,
            {weights[1].tensor, relu ? nullptr : weights[0].tensor, weights[2].tensor});
        ggml_set_output(gpu_lanes);
        auto * expected = modules::MoeModule::combine(ref, gpu_lanes, mixing.tensor);
        ggml_set_output(expected);
        ggml_build_forward_expand(reference.graph(), expected);
        reference.allocate();
        std::unique_ptr<core::PreparedGraph> host_reference;
        ggml_tensor * host_hidden = nullptr, * host_ids = nullptr, * host_lanes = nullptr;
        if (quantized && execution.moe_cpu_share() > 0) {
            host_reference = std::make_unique<core::PreparedGraph>(host,
                core::PreparedGraphConfig{"test.moe.auto.cpu_reference", 4 * 1024 * 1024, 256, 16 * ggml_tensor_overhead()});
            auto * persistent = host_reference->persistent_build_context().ggml;
            host_hidden = ggml_new_tensor_3d(persistent, GGML_TYPE_F32, 256, 1, 3);
            host_ids = ggml_new_tensor_2d(persistent, GGML_TYPE_I32, 2, 3);
            ggml_set_input(host_hidden);
            ggml_set_input(host_ids);
            host_reference->allocate_persistent();
            host_lanes = module.build_experts(host_reference->build_context(), host_hidden, host_ids,
                {store.host_tensor(group, store.tensor(group, 1).tensor).tensor,
                 relu ? nullptr : store.host_tensor(group, store.tensor(group, 0).tensor).tensor,
                 store.host_tensor(group, store.tensor(group, 2).tensor).tensor});
            ggml_set_output(host_lanes);
            ggml_build_forward_expand(host_reference->graph(), host_lanes);
            host_reference->allocate();
        }
        if (!core::DiskExpertWeightStore::evaluate(bank_ids, true, &store))
            throw std::runtime_error("automatic MoE modified ordinary routing evaluation");
        for (int request = 0; request < 8; ++request) {
            if (request < 4) {
                store.invalidate_cache();
                store.stage(group, {0, 1, 2, 3});
            } else if (request >= 6) store.stage(group, {4, 5, 6, 7});
            std::vector<float> values(3 * 256);
            for (size_t i = 0; i < values.size(); ++i) values[i] = (quantized ? 0.1F : 0.01F) * std::sin(float(i) * 0.07F + float(request));
            const std::vector<int32_t> ids{4, 5, 6, 7, 4, 7};
            core::write_tensor_f32(hidden, values);
            ggml_backend_tensor_set(logical.tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
            core::write_tensor_f32(mixing, {0.3F, 0.7F, 0.6F, 0.4F, 0.2F, 0.8F});
            const auto before = store.statistics().expert_loads;
            core::write_tensor_f32(counter, {0.0F});
            core::write_tensor_f32(one, {1.0F});
            owner.compute();
            std::vector<int32_t> observed_ids(ids.size());
            ggml_backend_tensor_get(logical.tensor, observed_ids.data(), 0, observed_ids.size() * sizeof(int32_t));
            if (observed_ids != ids) throw std::runtime_error("automatic MoE overwrote logical input IDs");
            if (core::read_tensor_f32(counter.tensor)[0] != 1.0F)
                throw std::runtime_error("hybrid boundary replayed upstream state updates");
            reference.compute();
            const auto actual = core::read_tensor_f32(output);
            auto baseline = core::read_tensor_f32(expected);
            if (host_reference) {
                ggml_backend_tensor_set(host_hidden, values.data(), 0, values.size() * sizeof(float));
                ggml_backend_tensor_set(host_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
                host_reference->compute();
                const auto cpu = core::read_tensor_f32(host_lanes), gpu = core::read_tensor_f32(gpu_lanes);
                const float share = execution.moe_cpu_share();
                // This fixture has assignment counts 4:2, 7:2, 5:1, 6:1.
                // At half share the cold misses send {7,6} to CPU, then {6},
                // then none as the remaining misses become GPU hits. At full
                // share all misses stay on CPU until the explicit warm-hit run.
                const auto cpu_id = [&](int32_t id) {
                    if (request >= 6) return false;
                    if (share == 1) return true;
                    if (share != 0.5F) throw std::runtime_error("quantized fixture requires share 0, 0.5 or 1");
                    return request < 4 ? id == 7 || id == 6 : request == 4 && id == 6;
                };
                const std::vector<float> scales{0.3F, 0.7F, 0.6F, 0.4F, 0.2F, 0.8F};
                for (size_t token = 0; token < 3; ++token) for (size_t channel = 0; channel < 256; ++channel) {
                    float sum = 0;
                    for (size_t lane = 0; lane < 2; ++lane) {
                        const size_t assignment = token * 2 + lane;
                        const auto & source = cpu_id(ids[assignment]) ? cpu : gpu;
                        sum += source[assignment * 256 + channel] * scales[assignment];
                    }
                    baseline[token * 256 + channel] = sum;
                }
            }
            float maximum = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error("automatic hybrid produced nonfinite output");
                maximum = std::max(maximum, std::abs(actual[i] - baseline[i]));
            }
            const auto gpu_loads = store.statistics().expert_loads - before;
            const auto wanted = execution.moe_cpu_share() == 0 ? 4 : execution.moe_cpu_share() == 1 ? 0 : 2;
            if (maximum > 1e-4F || (request < 4 && gpu_loads != static_cast<uint64_t>(wanted)) ||
                (request >= 6 && gpu_loads != 0))
                throw std::runtime_error("automatic hybrid changed expert math or uploaded CPU misses: " + std::to_string(maximum) + " request=" + std::to_string(request) + " loads=" + std::to_string(gpu_loads));
            std::cout << "automatic_moe quantized=" << quantized << " relu=" << relu << " rounded=" << rounded << " request=" << request
                      << " max_abs=" << maximum << " gpu_loads=" << gpu_loads << '\n';
        }
        reference.release();
        owner.release();
        if (owner.build_context().region_owner) throw std::runtime_error("released graph retained context ownership");
    }
}

// Two quantized expert regions share one allocation and full/body entry points.
// Resident native CPU/GPU graphs provide lane oracles; no region or disk staging
// participates in the reference. The expected placements are fixed fixtures.
void automatic_hybrid_lifecycle(engine::core::ExecutionContext & execution, engine::core::ExecutionContext & host,
    engine::core::DiskExpertWeightStore & store, size_t first_group,
    const std::vector<engine::core::TensorValue> & weights) {
    namespace core = engine::core;
    namespace modules = engine::modules;
    const float share = execution.moe_cpu_share();
    if (share != 0.5F && share != 1) return;
    constexpr size_t h = 256, tokens = 3, selected = 2, rows = tokens * selected;
    const std::vector<std::string> names{store.tensor(first_group, 0).tensor->name,
        store.tensor(first_group, 1).tensor->name, store.tensor(first_group, 2).tensor->name};
    const size_t second_group = store.add_group(names, execution.backend(), 4);
    modules::MoeModule module(modules::MoeExpertConfig{});
    std::vector<float> repeated;
    size_t workspace_bytes = 0;
    for (int cycle = 0; cycle < 2; ++cycle) {
        core::PreparedGraph owner(execution, {"test.moe.auto.lifecycle", 8 * 1024 * 1024, 512, 32 * ggml_tensor_overhead()});
        auto & persistent = owner.persistent_build_context();
        auto hidden = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({tokens, h}));
        auto ids1 = core::make_tensor(persistent, GGML_TYPE_I32, core::TensorShape::from_dims({tokens, selected}));
        auto ids2 = core::make_tensor(persistent, GGML_TYPE_I32, core::TensorShape::from_dims({tokens, selected}));
        auto mixing = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({tokens, selected}));
        auto counter = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        auto one = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        for (auto tensor : {hidden, ids1, ids2, mixing, counter, one}) ggml_set_input(tensor.tensor);
        auto & ctx = owner.build_context();
        auto * advance = ggml_cpy(ctx.ggml, ggml_add(ctx.ggml, counter.tensor, one.tensor), counter.tensor);
        auto * conditioning = ggml_add(ctx.ggml, ggml_scale(ctx.ggml, hidden.tensor, 0.75F), ggml_scale(ctx.ggml, advance, 0));
        ggml_set_output(conditioning);
        ggml_build_forward_expand(owner.graph(), conditioning);
        auto * output1 = module.build(ctx, conditioning, {ids1.tensor, mixing.tensor},
            {store.tensor(first_group, 1).tensor, store.tensor(first_group, 0).tensor, store.tensor(first_group, 2).tensor},
            store.route(ctx, ids1.tensor, first_group));
        auto * middle = ggml_scale(ctx.ggml, output1, 0.5F);
        auto * output2 = module.build(ctx, middle, {ids2.tensor, mixing.tensor},
            {store.tensor(second_group, 1).tensor, store.tensor(second_group, 0).tensor, store.tensor(second_group, 2).tensor},
            store.route(ctx, ids2.tensor, second_group));
        ggml_set_output(output1);
        // Retain the backing sum as well as its final reshape for the later check.
        ggml_set_output(output1->view_src);
        ggml_set_output(output2);
        ggml_build_forward_expand(owner.graph(), output2);

        struct Reference {
            std::unique_ptr<core::PreparedGraph> graph;
            ggml_tensor * hidden, * ids, * output;
        };
        std::vector<Reference> references;
        for (auto * backend : {&execution, &host}) {
            Reference ref;
            ref.graph = std::make_unique<core::PreparedGraph>(*backend,
                core::PreparedGraphConfig{"test.moe.lifecycle.reference", 4 * 1024 * 1024, 256, 16 * ggml_tensor_overhead()});
            auto * meta = ref.graph->persistent_build_context().ggml;
            ref.hidden = ggml_new_tensor_3d(meta, GGML_TYPE_F32, h, 1, tokens);
            ref.ids = ggml_new_tensor_2d(meta, GGML_TYPE_I32, selected, tokens);
            ggml_set_input(ref.hidden);
            ggml_set_input(ref.ids);
            ref.graph->allocate_persistent();
            const bool cpu = backend == &host;
            ref.output = module.build_experts(ref.graph->build_context(), ref.hidden, ref.ids,
                {cpu ? store.host_tensor(first_group, store.tensor(first_group, 1).tensor).tensor : weights[1].tensor,
                 cpu ? store.host_tensor(first_group, store.tensor(first_group, 0).tensor).tensor : weights[0].tensor,
                 cpu ? store.host_tensor(first_group, store.tensor(first_group, 2).tensor).tensor : weights[2].tensor});
            ggml_set_output(ref.output);
            ggml_build_forward_expand(ref.graph->graph(), ref.output);
            ref.graph->allocate();
            references.push_back(std::move(ref));
        }
        const std::vector<float> scales{0.3F, 0.7F, 0.6F, 0.4F, 0.2F, 0.8F};
        const auto expected_output = [&](const std::vector<float> & input, const std::vector<int32_t> & ids,
            int base, bool mixed) {
            for (auto & ref : references) {
                ggml_backend_tensor_set(ref.hidden, input.data(), 0, input.size() * sizeof(float));
                ggml_backend_tensor_set(ref.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
                ref.graph->compute();
            }
            const auto gpu = core::read_tensor_f32(references[0].output), cpu = core::read_tensor_f32(references[1].output);
            std::vector<float> result(h * tokens, 0);
            for (size_t row = 0; row < rows; ++row) {
                const bool cpu_lane = mixed ? share == 1 && ids[row] == 0 :
                    share == 1 || ids[row] == base + 2 || ids[row] == base + 3;
                const auto & source = cpu_lane ? cpu : gpu;
                for (size_t channel = 0; channel < h; ++channel)
                    result[(row / selected) * h + channel] += source[row * h + channel] * scales[row];
            }
            return result;
        };
        const auto check_output = [&](ggml_tensor * output, const std::vector<float> & expected, const char * layer) {
            const auto actual = core::read_tensor_f32(output);
            float maximum = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error("automatic quantized lifecycle produced nonfinite output");
                maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
            }
            if (maximum > 1e-4F) throw std::runtime_error(std::string("automatic quantized lifecycle ") + layer + " changed native expert math: " + std::to_string(maximum));
        };
        owner.allocate_persistent();
        auto * scheduler = owner.create_scheduler({execution.backend(), host.backend()});
        struct Observed {
            core::DiskExpertWeightStore * store;
            ggml_tensor * conditioning, * first;
            std::vector<float> input, output;
        } observed{&store, conditioning, output1, {}, {}};
        ggml_backend_sched_set_eval_callback(scheduler, [](ggml_tensor * node, bool ask, void * data) {
            auto & state = *static_cast<Observed *>(data);
            if (node == state.conditioning || node == state.first) {
                if (!ask) (node == state.conditioning ? state.input : state.output) = core::read_tensor_f32(node);
                return true;
            }
            return core::DiskExpertWeightStore::evaluate(node, ask, state.store);
        }, &observed);
        owner.allocate();
        if (cycle && owner.workspace_bytes() != workspace_bytes)
            throw std::runtime_error("automatic lifecycle changed workspace size after reprepare");
        workspace_bytes = owner.workspace_bytes();
        core::write_tensor_f32(counter, {0});
        core::write_tensor_f32(one, {1});
        core::write_tensor_f32(mixing, scales);
        std::vector<float> conditioned(h * tokens);
        for (int request = 0; request < 6; ++request) {
            const bool mixed = request == 2 || request == 3;
            const int base1 = request % 2 ? 8 : 4, base2 = request % 2 ? 4 : 8;
            const std::vector<int32_t> a = mixed ? std::vector<int32_t>{0, 8, 9, 10, 11, 0} :
                std::vector<int32_t>{base1, base1 + 1, base1 + 2, base1 + 3, base1, base1 + 3};
            const std::vector<int32_t> b = mixed ? std::vector<int32_t>{0, 4, 5, 6, 7, 0} :
                std::vector<int32_t>{base2, base2 + 1, base2 + 2, base2 + 3, base2, base2 + 3};
            store.invalidate_cache();
            store.stage(first_group, mixed ? std::vector<int32_t>{8, 9, 10, 11} : std::vector<int32_t>{0, 1, 2, 3});
            store.stage(second_group, mixed ? std::vector<int32_t>{4, 5, 6, 7} : std::vector<int32_t>{0, 1, 2, 3});
            std::vector<float> values(h * tokens);
            for (size_t i = 0; i < values.size(); ++i) values[i] = 0.1F * std::sin(float(i) * 0.07F + float(request % 4));
            for (size_t i = 0; i < values.size(); ++i) conditioned[i] = values[i] * 0.75F;
            core::write_tensor_f32(hidden, values);
            ggml_backend_tensor_set(ids1.tensor, a.data(), 0, a.size() * sizeof(int32_t));
            ggml_backend_tensor_set(ids2.tensor, b.data(), 0, b.size() * sizeof(int32_t));
            const auto loads = store.statistics().expert_loads;
            owner.compute();
            const uint64_t wanted = share == 1 ? 0 : mixed ? 2 : 4;
            if (store.statistics().expert_loads - loads != wanted)
                throw std::runtime_error("automatic lifecycle evicted a future hit or uploaded a CPU miss");
            if (observed.input != conditioned) throw std::runtime_error("automatic lifecycle changed conditioning input");
            if (observed.output != core::read_tensor_f32(output1))
                throw std::runtime_error("automatic lifecycle overwrote the first layer output");
            auto expected1 = expected_output(conditioned, a, base1, mixed);
            check_output(output1, expected1, "first");
            for (auto & value : expected1) value *= 0.5F;
            check_output(output2, expected_output(expected1, b, base2, mixed), "second");
            if (core::read_tensor_f32(counter.tensor)[0] != static_cast<float>(request + 1))
                throw std::runtime_error("automatic lifecycle skipped or replayed conditioning state");
            if (request == 0 && cycle == 0) repeated = core::read_tensor_f32(output2);
            if (request == 4 || (request == 0 && cycle == 1))
                if (core::read_tensor_f32(output2) != repeated) throw std::runtime_error("automatic lifecycle repeated request drifted");
            std::cout << "automatic_lifecycle cycle=" << cycle << " request=" << request << " mixed=" << mixed << '\n';
        }
        owner.reset();
    }
}

void native_hybrid_lanes(engine::core::ExecutionContext & execution, engine::core::ExecutionContext & host,
    engine::core::DiskExpertWeightStore & store, size_t group, const std::vector<engine::core::TensorValue> & weights) {
    namespace core = engine::core;
    namespace modules = engine::modules;
    constexpr int64_t hidden_size = 256, tokens = 3, selected = 2, rows = tokens * selected;
    modules::MoeModule module(modules::MoeExpertConfig{});
    auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
    modules::StagedMoeConfig config;
    config.hidden_size = hidden_size;
    config.tokens = tokens;
    config.selected = selected;
    config.experts = 12;
    config.cache_slots = 4;
    config.rows_per_expert = {1, 2, 4};
    modules::StagedMoeRuntime hybrid(execution, config, workspace, &host);
    const auto layer = hybrid.add_layer(module, {store.tensor(group, 1).tensor,
        store.tensor(group, 0).tensor, store.tensor(group, 2).tensor}, store, group);
    core::PreparedGraph inputs(execution, {"test.native_hybrid.inputs", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
    auto * ctx = inputs.persistent_build_context().ggml;
    auto * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected, tokens);
    auto * mixing = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, selected, tokens);
    for (auto * tensor : {hidden, ids, mixing}) ggml_set_input(tensor);
    inputs.allocate_persistent();
    hybrid.build_inputs(inputs.build_context(), inputs.graph(), hidden, {ids, mixing});
    workspace->add(inputs);
    workspace->prepare();
    hybrid.prepare();
    core::PreparedGraph gpu_reference(execution, {"test.native_hybrid.gpu", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
    core::PreparedGraph cpu_reference(host, {"test.native_hybrid.cpu", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
    std::vector<ggml_tensor *> reference_inputs, reference_ids, reference_outputs;
    for (auto * graph : {&gpu_reference, &cpu_reference}) {
        auto & build = graph->build_context();
        auto * persistent = graph->persistent_build_context().ggml;
        auto * values = ggml_new_tensor_3d(persistent, GGML_TYPE_F32, hidden_size, 1, rows);
        auto * logical = ggml_new_tensor_2d(persistent, GGML_TYPE_I32, 1, rows);
        ggml_set_input(values);
        ggml_set_input(logical);
        graph->allocate_persistent();
        const bool cpu = graph == &cpu_reference;
        const modules::MoeExpertWeights bank{
            cpu ? store.host_tensor(group, store.tensor(group, 1).tensor).tensor : weights[1].tensor,
            cpu ? store.host_tensor(group, store.tensor(group, 0).tensor).tensor : weights[0].tensor,
            cpu ? store.host_tensor(group, store.tensor(group, 2).tensor).tensor : weights[2].tensor};
        auto * output = module.build_experts(build, values, logical, bank);
        ggml_set_output(output);
        ggml_build_forward_expand(graph->graph(), output);
        graph->allocate();
        reference_inputs.push_back(values);
        reference_ids.push_back(logical);
        reference_outputs.push_back(output);
    }
    const std::vector<int32_t> logical{4, 5, 6, 7, 4, 7};
    const std::vector<float> scales(rows, 0.5F);
    for (int request = 0; request < 4; ++request) {
        store.invalidate_cache();
        store.stage(group, {0, 1, 2, 3});
        std::vector<float> values(hidden_size * tokens), expanded(hidden_size * rows);
        for (size_t i = 0; i < values.size(); ++i) values[i] = 0.1F * std::sin(float(i) * 0.07F + float(request));
        for (int64_t row = 0; row < rows; ++row)
            std::copy_n(values.begin() + (row / selected) * hidden_size, hidden_size, expanded.begin() + row * hidden_size);
        ggml_backend_tensor_set(hidden, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(ids, logical.data(), 0, logical.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mixing, scales.data(), 0, scales.size() * sizeof(float));
        inputs.compute();
        hybrid.run(layer);
        const auto actual = core::read_tensor_f32(hybrid.output());
        for (size_t i = 0; i < reference_inputs.size(); ++i) {
            ggml_backend_tensor_set(reference_inputs[i], expanded.data(), 0, expanded.size() * sizeof(float));
            ggml_backend_tensor_set(reference_ids[i], logical.data(), 0, logical.size() * sizeof(int32_t));
        }
        gpu_reference.compute();
        cpu_reference.compute();
        const auto gpu = core::read_tensor_f32(reference_outputs[0]), cpu = core::read_tensor_f32(reference_outputs[1]);
        float maximum = 0, native_difference = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            // Stable descending assignment counts give GPU {4,5}, CPU {7,6}.
            const bool on_cpu = logical[i / hidden_size] == 7 || logical[i / hidden_size] == 6;
            maximum = std::max(maximum, std::abs(actual[i] - (on_cpu ? cpu[i] : gpu[i])));
            native_difference = std::max(native_difference, std::abs(cpu[i] - gpu[i]));
            if (!std::isfinite(actual[i])) throw std::runtime_error("native hybrid expert output is nonfinite");
        }
        std::cout << "native_hybrid request=" << request << " placement_max_abs=" << maximum
                  << " cpu_gpu_max_abs=" << native_difference << '\n';
        if (maximum > 1e-4F) throw std::runtime_error("hybrid expert lanes differ from their native backend reference");
    }
}


// The CPU callback rendezvous occurs inside expert graph execution, including
// the hybrid worker on accelerators. Cancellation releases a waiting peer if
// setup, computation or an oracle assertion fails in either session.
struct ConcurrentMoeOverlap {
    std::mutex mutex;
    std::condition_variable changed;
    bool cancelled = false;
    size_t waiting = 0, generation = 0;
    std::array<ggml_backend_t, 2> primary{}, cpu{};
    std::array<engine::core::DiskExpertWeightStore *, 2> stores{};
    std::array<engine::core::MoeResources *, 2> resources{};
    std::array<ggml_tensor *, 2> banks{};
    std::array<ggml_backend_buffer_t, 2> buffers{};
    struct Participant {
        ConcurrentMoeOverlap & barrier;
        size_t session;
        std::atomic<bool> armed{false};
    };
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = true;
        changed.notify_all();
    }
    static bool expert_callback(void * data) noexcept {
        auto & participant = *static_cast<Participant *>(data);
        if (!participant.armed.exchange(false)) return false;
        auto & barrier = participant.barrier;
        try {
            std::unique_lock<std::mutex> lock(barrier.mutex);
            if (barrier.cancelled) return true;
            const auto generation = barrier.generation;
            if (++barrier.waiting == 2) {
                barrier.waiting = 0;
                ++barrier.generation;
                barrier.changed.notify_all();
            } else if (!barrier.changed.wait_for(lock, std::chrono::seconds(30), [&] {
                return barrier.cancelled || barrier.generation != generation;
            })) {
                barrier.cancelled = true;
                barrier.changed.notify_all();
            }
            return barrier.cancelled;
        } catch (...) {
            barrier.cancel();
            return true;
        }
    }
};

void dense_hybrid_oracle(engine::core::ExecutionContext & execution, engine::core::DiskExpertWeightStore & store,
    size_t group, const std::vector<engine::io::SafeTensorWriteEntry> & entries, bool cpu_staged = false, bool pinned_cpu = false,
    ConcurrentMoeOverlap::Participant * overlap = nullptr) {
    namespace core = engine::core;
    namespace modules = engine::modules;
    constexpr int64_t h = 256, tokens = 3, selected = 2;
    std::vector<std::vector<float>> weights;
    for (const auto & entry : entries) {
        weights.emplace_back(entry.data.size() / sizeof(float));
        std::memcpy(weights.back().data(), entry.data.data(), entry.data.size());
    }
    for (bool relu : {false, true}) for (bool rounded : {false, true}) {
        auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
        modules::StagedMoeConfig config;
        config.hidden_size = h;
        config.tokens = tokens;
        config.selected = selected;
        config.experts = 12;
        config.cache_slots = 4;
        config.rows_per_expert = {1, 2, 4};
        modules::StagedMoeRuntime runtime(execution, config, workspace, store.cpu_execution());
        modules::MoeExpertConfig math;
        math.activation = relu ? modules::MoeActivation::ReluSquared : modules::MoeActivation::SwiGlu;
        math.round_projections = math.round_activation = math.round_product = math.round_output = rounded;
        const auto layer = runtime.add_layer(modules::MoeModule(math),
            {store.tensor(group, 1).tensor, relu ? nullptr : store.tensor(group, 0).tensor,
             store.tensor(group, 2).tensor}, store, group);
        core::PreparedGraph graph(execution, {"test.hybrid_moe_dense", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
        auto * inputs = graph.persistent_build_context().ggml;
        auto * hidden = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, h, tokens);
        auto * logical = ggml_new_tensor_2d(inputs, GGML_TYPE_I32, selected, tokens);
        auto * mixing = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, selected, tokens);
        for (auto * tensor : {hidden, logical, mixing}) ggml_set_input(tensor);
        graph.allocate_persistent();
        auto & ctx = graph.build_context();
        runtime.build_inputs(ctx, graph.graph(), hidden, {logical, mixing});
        const auto finish = graph.add_graph("finish", 128);
        auto * reduced = modules::MoeModule::combine(ctx, runtime.output(), runtime.routing_weights());
        ggml_set_output(reduced);
        ggml_build_forward_expand(graph.graph(finish), reduced);
        workspace->add(graph);
        workspace->prepare();
        runtime.prepare();
        std::vector<float> values(static_cast<size_t>(h * tokens));
        const std::vector<float> mixing_values{0.3F, 0.7F, 0.6F, 0.4F, 0.2F, 0.8F};
        const auto round = [rounded](float value) {
            return rounded ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)) : value;
        };
        for (int request = 0; request < 4; ++request) {
            store.stage(group, {0, 1, 2, 3});
            const bool high_experts = (request % 2 != 0) != (overlap && overlap->session == 1);
            const std::vector<int32_t> ids = high_experts ? std::vector<int32_t>{8, 9, 10, 11, 8, 11}
                : std::vector<int32_t>{4, 5, 6, 7, 4, 7};
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = 0.01F * std::sin(float(i) * 0.07F + float(request) +
                    (overlap ? 0.37F * float(overlap->session) : 0.0F));
            ggml_backend_tensor_set(hidden, values.data(), 0, values.size() * sizeof(float));
            ggml_backend_tensor_set(logical, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(mixing, mixing_values.data(), 0, mixing_values.size() * sizeof(float));
            graph.compute();
            if (overlap) overlap->armed.store(true);
            runtime.run(layer);
            if (overlap && overlap->armed.exchange(false))
                throw std::runtime_error("concurrent MoE never entered CPU expert computation");
            const auto lanes = core::read_tensor_f32(runtime.output());
            std::vector<float> expected(lanes.size());
            for (int64_t t = 0; t < tokens; ++t) for (int64_t lane = 0; lane < selected; ++lane) {
                const auto expert = ids[static_cast<size_t>(t * selected + lane)];
                std::vector<float> activation(static_cast<size_t>(h));
                for (int64_t row = 0; row < h; ++row) {
                    double up = 0, gate = 0;
                    for (int64_t col = 0; col < h; ++col) {
                        const auto index = static_cast<size_t>((expert * h + row) * h + col);
                        up += double(weights[1][index]) * values[static_cast<size_t>(t * h + col)];
                        gate += double(weights[0][index]) * values[static_cast<size_t>(t * h + col)];
                    }
                    const float u = round(static_cast<float>(up)), g = round(static_cast<float>(gate));
                    const float a = round(relu ? std::max(u, 0.0F) : g / (1.0F + std::exp(-g)));
                    activation[static_cast<size_t>(row)] = round(a * (relu ? a : u));
                }
                for (int64_t row = 0; row < h; ++row) {
                    double value = 0;
                    for (int64_t col = 0; col < h; ++col)
                        value += double(weights[2][static_cast<size_t>((expert * h + row) * h + col)]) * activation[static_cast<size_t>(col)];
                    expected[static_cast<size_t>((t * selected + lane) * h + row)] = round(static_cast<float>(value));
                }
            }
            graph.compute(finish);
            const auto actual = core::read_tensor_f32(reduced);
            float maximum = 0;
            for (size_t i = 0; i < lanes.size(); ++i) maximum = std::max(maximum, std::abs(lanes[i] - expected[i]));
            for (int64_t t = 0; t < tokens; ++t) for (int64_t row = 0; row < h; ++row) {
                float value = 0;
                for (int64_t lane = 0; lane < selected; ++lane)
                    value += expected[static_cast<size_t>((t * selected + lane) * h + row)] * mixing_values[static_cast<size_t>(t * selected + lane)];
                maximum = std::max(maximum, std::abs(actual[static_cast<size_t>(t * h + row)] - value));
            }
            if (maximum > 1e-4F || !std::all_of(lanes.begin(), lanes.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("dense hybrid MoE disagrees with independent expert math: " + std::to_string(maximum));
            std::cout << "dense hybrid relu=" << relu << " rounded=" << rounded << " request=" << request << " max_abs=" << maximum << '\n';
        }
        if (cpu_staged) {
            if (!store.cpu_execution() || store.group_backend(group) != store.cpu_execution()->backend() ||
                (pinned_cpu ? store.statistics().host_buffer_bytes == 0 : store.statistics().host_buffer_bytes != 0) ||
                runtime.statistics().gpu_assignments != 0 ||
                (store.group_backend(group) != execution.backend() && runtime.statistics().cpu_assignments != uint64_t(4 * tokens * selected)))
                throw std::runtime_error("bounded CPU oracle did not execute every assignment on the cache backend");
        } else {
            if ((execution.moe_cpu_share() > 0 && runtime.statistics().cpu_assignments == 0) ||
                (execution.moe_cpu_share() < 1 && runtime.statistics().gpu_assignments == 0 && execution.moe_cpu_share() > 0))
                throw std::runtime_error("dense hybrid test did not execute both placements");
        }
    }
}

void moe_policy_resolution() {
    namespace core = engine::core;
    using Options = std::unordered_map<std::string, std::string>;
    const auto same = [](const core::MoeExecutionPolicy & a, const core::MoeExecutionPolicy & b) {
        return a.residency == b.residency && a.pin_host_weights == b.pin_host_weights &&
            a.cache_slots == b.cache_slots && a.read_threads == b.read_threads &&
            a.cpu_layers == b.cpu_layers && a.cpu_share == b.cpu_share;
    };
    // Defaults belong to callers: test the existing model policies without
    // importing model code or requiring model packages.
    for (const std::string family : {"fish_drama3", "ming3_audio", "nemotron_transcribe"}) {
        for (bool metal : {false, true}) {
            core::MoeExecutionPolicy defaults;
            if (family == "fish_drama3") {
                defaults.cache_slots = 32;
                defaults.cpu_layers = {0, 1, 2, 3};
            }
            if (family == "ming3_audio") defaults.pin_host_weights = !metal;
            if (family == "nemotron_transcribe") defaults.read_threads = 4;
            if (!same(core::resolve_moe_execution_policy({}, family, defaults, 6), defaults))
                throw std::runtime_error("MoE resolver changed model-provided defaults: " + family);
            const std::vector<std::tuple<std::string, std::string, std::string>> aliases{
                {"moe_disk_experts", "disk_experts", "true"},
                {"moe_pin_experts", "pin_experts", "true"},
                {"moe_expert_cache_size", "expert_cache_size", "4"},
                {"moe_disk_expert_read_threads", "disk_expert_read_threads", "2"},
                {"moe_cpu_expert_layers", "cpu_expert_layers", "2"}};
            for (const auto & [shared, local, value] : aliases) {
                const auto legacy = family + "." + local;
                const auto common_policy = core::resolve_moe_execution_policy({{shared, value}}, family, defaults, 6);
                const auto legacy_policy = core::resolve_moe_execution_policy({{legacy, value}}, family, defaults, 6);
                if (!same(common_policy, legacy_policy))
                    throw std::runtime_error("MoE common/legacy policies differ: " + shared);
                // Equal values are still ambiguous, just like conflicting ones.
                for (const auto & other : {value, local == "disk_experts" || local == "pin_experts"
                        ? std::string("false") : std::string("1")}) {
                    bool rejected = false;
                    try { core::resolve_moe_execution_policy({{shared, value}, {legacy, other}}, family, defaults, 6); }
                    catch (const std::invalid_argument &) { rejected = true; }
                    if (!rejected) throw std::runtime_error("MoE duplicate option spellings accepted: " + shared);
                }
            }
            const auto combined = core::resolve_moe_execution_policy({
                {"moe_disk_experts", "true"}, {family + ".pin_experts", "false"},
                {"moe_expert_cache_size", "4"}, {family + ".disk_expert_read_threads", "2"},
                {"moe_cpu_expert_layers", "2"}, {"moe_cpu_share", "0.5"}}, family, defaults, 6);
            if (combined.residency != core::MoeExpertResidency::Staged || combined.pin_host_weights ||
                combined.cache_slots != 4 || combined.read_threads != 2 || combined.cpu_share != 0.5F ||
                combined.cpu_layers != std::vector<size_t>({0, 1}))
                throw std::runtime_error("MoE resolver ignored an explicit option");
            const auto count = core::resolve_moe_execution_policy({{"moe_cpu_expert_layers", "2"}}, family, defaults, 6);
            const auto zero = core::resolve_moe_execution_policy({{family + ".cpu_expert_layers", "0"}}, family, defaults, 6);
            if (count.cpu_layers != std::vector<size_t>({0, 1}) || !zero.cpu_layers.empty())
                throw std::runtime_error("legacy MoE CPU count did not select leading MoE ordinals");
        }
    }
    core::MoeExecutionPolicy explicit_layers;
    explicit_layers.cpu_layers = {3, 1};
    if (core::resolve_moe_execution_policy({}, "test", explicit_layers, 4).cpu_layers != explicit_layers.cpu_layers)
        throw std::runtime_error("MoE resolver changed explicit CPU ordinals");
    for (const auto & ordinals : {std::vector<size_t>{1, 1}, std::vector<size_t>{4}}) {
        auto invalid = explicit_layers;
        invalid.cpu_layers = ordinals;
        bool rejected = false;
        try { core::resolve_moe_execution_policy({}, "test", invalid, 4); }
        catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid explicit CPU ordinals accepted");
    }
    const std::vector<Options> invalid_options{
        {{"moe_cpu_expert_layers", "-1"}}, {{"test.cpu_expert_layers", "5"}},
        {{"moe_expert_cache_size", "0"}}, {{"test.expert_cache_size", "-1"}},
        {{"moe_disk_expert_read_threads", "0"}}, {{"test.disk_expert_read_threads", "65"}},
        {{"moe_cpu_share", "-0.1"}}, {{"moe_cpu_share", "1.1"}},
        {{"moe_cpu_share", "nan"}}, {{"moe_cpu_share", "inf"}}, {{"moe_cpu_share", "0.5"}},
        {{"moe_disk_experts", "not-a-bool"}}, {{"test.pin_experts", "not-a-bool"}}};
    for (const auto & options : invalid_options) {
        bool rejected = false;
        try { core::resolve_moe_execution_policy(options, "test", {}, 4); }
        catch (const std::invalid_argument &) { rejected = true; }
        catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid active MoE policy accepted");
    }
    for (const auto & share : {"0", "0.5", "1"}) {
        const auto policy = core::resolve_moe_execution_policy(
            {{"moe_disk_experts", "true"}, {"moe_cpu_share", share}}, "test", explicit_layers, 4);
        if (policy.residency != core::MoeExpertResidency::Staged || policy.cpu_share != std::stof(share) ||
            policy.cpu_layers != explicit_layers.cpu_layers)
            throw std::runtime_error("staged MoE share changed whole-layer selection");
    }
}

// Exercise the primary-backend input/combine graphs around a CPU-owned bounded
// cache. A CPU primary also covers reuse of the existing session context.
void bounded_cpu_staged_execution(engine::core::ExecutionContext & execution,
    engine::core::ExecutionContext & cpu, engine::core::DiskExpertWeightStore & store, size_t group,
    const std::vector<engine::core::TensorValue> & resident, bool pinned = false) {
    namespace core = engine::core;
    namespace modules = engine::modules;
    constexpr int64_t h = 256, tokens = 7, selected = 2;
    if (store.group_backend(group) != cpu.backend() || store.cpu_execution() != &cpu ||
        !core::is_host_backend(store.group_backend(group)))
        throw std::runtime_error("bounded CPU execution did not borrow the cache's CPU context");
    const auto before = store.statistics();
    size_t native_bytes = 0, cache_bytes = 0;
    for (size_t projection = 0; projection < resident.size(); ++projection) {
        native_bytes += ggml_nbytes(resident[projection].tensor);
        cache_bytes += ggml_nbytes(store.tensor(group, projection).tensor);
    }
    if (!before.buffer_bytes || !cache_bytes || cache_bytes >= native_bytes ||
        (pinned ? before.host_buffer_bytes < native_bytes : before.host_buffer_bytes != 0))
        throw std::runtime_error("CPU cache residency does not match explicit pinning policy");
    auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
    modules::StagedMoeConfig config;
    config.hidden_size = h;
    config.tokens = tokens;
    config.selected = selected;
    config.experts = 12;
    config.cache_slots = 4;
    config.rows_per_expert = {1, 2, 4};
    modules::MoeModule module(modules::MoeExpertConfig{});
    modules::StagedMoeRuntime runtime(execution, config, workspace, &cpu);
    const auto layer = runtime.add_layer(module, {store.tensor(group, 1).tensor,
        store.tensor(group, 0).tensor, store.tensor(group, 2).tensor}, store, group);
    core::PreparedGraph graph(execution, {"test.moe.bounded_cpu", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
    auto * meta = graph.persistent_build_context().ggml;
    auto * hidden = ggml_new_tensor_2d(meta, GGML_TYPE_F32, h, tokens);
    auto * ids = ggml_new_tensor_2d(meta, GGML_TYPE_I32, selected, tokens);
    auto * mixing = ggml_new_tensor_2d(meta, GGML_TYPE_F32, selected, tokens);
    for (auto * tensor : {hidden, ids, mixing}) ggml_set_input(tensor);
    graph.allocate_persistent();
    runtime.build_inputs(graph.build_context(), graph.graph(), hidden, {ids, mixing});
    const auto finish = graph.add_graph("combine", 128);
    auto * combined = modules::MoeModule::combine(graph.build_context(), runtime.output(), runtime.routing_weights());
    ggml_set_output(combined);
    ggml_build_forward_expand(graph.graph(finish), combined);
    workspace->add(graph);
    workspace->prepare();
    runtime.prepare();
    const auto workspace_bytes = runtime.workspace_bytes();
    // Reference banks were uploaded before source release. Do not call
    // host_tensor(): that would create full banks in the bounded-cache store.
    core::PreparedGraph reference(cpu, {"test.moe.bounded_cpu.reference", 1024 * 1024, 128, 16 * ggml_tensor_overhead()});
    auto * ref_meta = reference.persistent_build_context().ggml;
    auto * ref_input = ggml_new_tensor_3d(ref_meta, GGML_TYPE_F32, h, 1, tokens);
    auto * ref_ids = ggml_new_tensor_2d(ref_meta, GGML_TYPE_I32, selected, tokens);
    ggml_set_input(ref_input);
    ggml_set_input(ref_ids);
    reference.allocate_persistent();
    auto * ref_output = module.build_experts(reference.build_context(), ref_input, ref_ids,
        {resident[1].tensor, resident[0].tensor, resident[2].tensor});
    ggml_set_output(ref_output);
    ggml_build_forward_expand(reference.graph(), ref_output);
    reference.allocate();
    store.invalidate_cache();
    uint64_t previous_loads = store.statistics().expert_loads;
    for (int request = 0; request < 5; ++request) {
        std::vector<float> values(h * tokens), scales(tokens * selected);
        std::vector<int32_t> routes(tokens * selected);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = 0.01F * std::sin(float(i) * 0.07F + float(request == 1 ? 0 : request));
        for (size_t i = 0; i < routes.size(); ++i) {
            routes[i] = request == 2 ? static_cast<int32_t>(i % 12)
                : static_cast<int32_t>((request == 3 ? 8 : 0) + i % 4);
            scales[i] = i % 2 ? 0.7F : 0.3F;
        }
        ggml_backend_tensor_set(hidden, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mixing, scales.data(), 0, scales.size() * sizeof(float));
        graph.compute();
        // Cover both public run entry points with identical assignment ordering.
        if (request % 2) runtime.run(layer, routes);
        else runtime.run(layer);
        graph.compute(finish);
        ggml_backend_tensor_set(ref_input, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(ref_ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        reference.compute();
        const auto expected = core::read_tensor_f32(ref_output);
        const auto lanes = core::read_tensor_f32(runtime.output());
        const auto result = core::read_tensor_f32(combined);
        if (expected.size() != size_t(h * tokens * selected) || lanes.size() != expected.size() ||
            result.size() != size_t(h * tokens))
            throw std::runtime_error("bounded CPU execution changed output geometry");
        for (size_t i = 0; i < lanes.size(); ++i)
            if (!std::isfinite(lanes[i]) || !std::isfinite(expected[i]) || std::abs(lanes[i] - expected[i]) > 1e-4F)
                throw std::runtime_error("bounded CPU expert lanes differ from native CPU math");
        for (int64_t token = 0; token < tokens; ++token) for (int64_t row = 0; row < h; ++row) {
            float value = 0;
            for (int64_t lane = 0; lane < selected; ++lane)
                value += expected[(token * selected + lane) * h + row] * scales[token * selected + lane];
            if (!std::isfinite(result[token * h + row]) || std::abs(result[token * h + row] - value) > 1e-4F)
                throw std::runtime_error("bounded CPU combine changed assignment/reduction order");
        }
        const auto stats = store.statistics();
        if (stats.buffer_bytes != before.buffer_bytes || stats.host_buffer_bytes != before.host_buffer_bytes ||
            (pinned && stats.bytes_read != before.bytes_read) ||
            runtime.workspace_bytes() != workspace_bytes ||
            (request == 1 && stats.expert_loads != previous_loads) ||
            (request == 2 && stats.expert_loads - previous_loads < 8))
            throw std::runtime_error("bounded CPU execution grew storage, missed warm reuse or failed to exercise eviction");
        previous_loads = stats.expert_loads;
        const auto & cached = store.cached_experts(group);
        if (cached.size() != 4) throw std::runtime_error("CPU cache exceeded four expert slots");
        for (size_t p = 0; p < resident.size(); ++p) {
            auto * bank = store.tensor(group, p).tensor;
            if (bank->type != resident[p].tensor->type || bank->ne[2] != 4)
                throw std::runtime_error("bounded CPU staging changed native bank type or size");
            const auto bytes = resident[p].tensor->nb[2];
            for (size_t slot = 0; slot < cached.size(); ++slot) {
                if (cached[slot] < 0 || cached[slot] >= 12)
                    throw std::runtime_error("bounded CPU staging left an invalid occupied cache slot");
                std::vector<std::byte> actual(bytes), native(bytes);
                ggml_backend_tensor_get(bank, actual.data(), slot * bytes, bytes);
                ggml_backend_tensor_get(resident[p].tensor, native.data(), size_t(cached[slot]) * bytes, bytes);
                if (actual != native) throw std::runtime_error("bounded CPU staging changed native expert bytes");
            }
        }
        const auto assignments = runtime.statistics();
        const auto expected_cpu = execution.backend() != cpu.backend() ? uint64_t((request + 1) * tokens * selected) : 0;
        if (assignments.cpu_assignments != expected_cpu || assignments.gpu_assignments != 0)
            throw std::runtime_error("whole-layer CPU staging redistributed assignments through hybrid placement");
    }
    std::cout << "bounded_cpu_staged type=" << ggml_type_name(resident[0].tensor->type)
              << " primary_cpu=" << core::is_host_backend(execution.backend())
              << " cpu_share=" << execution.moe_cpu_share() << " pinned=" << pinned << '\n';
}

void moe_resource_matrix(const std::filesystem::path & path, engine::core::ExecutionContext & requested_execution,
    const std::vector<std::string> & names, const std::vector<engine::core::TensorValue> & full,
    const std::vector<engine::io::SafeTensorWriteEntry> & entries) {
    namespace core = engine::core;
    namespace assets = engine::assets;
    std::vector<core::MoeExpertTensorBinding> bindings;
    for (const auto & name : names)
        bindings.push_back({name, name, {12, 256, 256}, assets::TensorStorageType::Native});
    const bool host_primary = core::is_host_backend(requested_execution.backend());
    // Each cycle reopens the same synthetic package after releasing the source
    // mapping. Nothing is borrowed across coordinator/store lifetimes.
    for (bool staged : {false, true}) for (bool cpu_layer : {false, true}) for (int cycle = 0; cycle < 2; ++cycle) {
        std::unique_ptr<core::ExecutionContext> resident_execution;
        if (!staged && requested_execution.moe_cpu_share() != 0)
            resident_execution = std::make_unique<core::ExecutionContext>(requested_execution.config());
        auto & execution = resident_execution ? *resident_execution : requested_execution;
        auto source = assets::open_tensor_source(path);
        core::BackendWeightStore primary(execution.backend(), execution.backend_type(), "test.moe.primary", 1024 * 1024);
        // The primary store owns backbone weights even when every expert bank
        // lives in a separate staged cache. Model that ownership explicitly.
        const std::vector<float> norm_values(256, 1.0F);
        const auto primary_norm = primary.make_f32(core::TensorShape::from_dims({256}), norm_values);
        core::MoeExecutionPolicy policy;
        policy.residency = staged ? core::MoeExpertResidency::Staged : core::MoeExpertResidency::Backend;
        policy.cache_slots = staged ? 4 : 8;
        policy.pin_host_weights = !staged && cycle == 1;
        if (cpu_layer) policy.cpu_layers = {1};
        policy.cpu_share = staged ? execution.moe_cpu_share() : 0;
        core::MoeResources resources(execution, *source, primary, policy, "test.moe.cpu", 1024 * 1024);
        const auto & first = resources.bind_layer(0, 2, bindings, 2);
        const auto & middle = resources.bind_layer(1, 5, bindings, 2);
        const auto & last = resources.bind_layer(2, 9, bindings, 2);
        if (&first != &resources.layer(0) || &middle != &resources.layer(1) || &last != &resources.layer(2) ||
            first.ordinal != 0 || middle.ordinal != 1 || last.ordinal != 2 ||
            first.backbone_index != 2 || middle.backbone_index != 5 || last.backbone_index != 9 ||
            first.cpu || middle.cpu != cpu_layer || last.cpu || resources.has_cpu_layers() != cpu_layer)
            throw std::runtime_error("MoE handles moved or confused backbone indices with CPU ordinals");
        if (bool(resources.disk_store()) != staged || resources.policy().cpu_layers != policy.cpu_layers)
            throw std::runtime_error("MoE resource policy/weight ownership changed");
        auto * cpu = resources.cpu_execution();
        if (host_primary ? cpu != &execution : bool(cpu) != (cpu_layer || policy.cpu_share > 0))
            throw std::runtime_error("MoE CPU context was eagerly allocated or primary CPU was not reused");
        auto & target = cpu_layer ? *cpu : execution;
        core::BackendWeightStore reference(target.backend(), target.backend_type(), "test.moe.reference", 1024 * 1024);
        std::vector<core::TensorValue> expected_weights;
        for (const auto & name : names)
            expected_weights.push_back(reference.load_tensor(*source, name, assets::TensorStorageType::Native, {12, 256, 256}));
        primary.upload();
        if (core::read_tensor_f32(primary_norm.tensor) != norm_values)
            throw std::runtime_error("MoE fixture did not upload primary backbone weights");
        reference.upload();
        resources.finalize_weights();
        source->release_storage();
        // A staged GPU coordinator has no routing CPU until a scheduler is
        // requested. Direct staged execution never needs to make this request.
        {
            core::PreparedGraph scheduler_owner(execution, {"test.moe.scheduler", 1024 * 1024, 128});
            if (bool(resources.configure_scheduler(scheduler_owner)) != (staged || (cpu_layer && !host_primary)))
                throw std::runtime_error("MoE scheduler isolation changed");
            auto * routing_cpu = resources.cpu_execution();
            if (host_primary ? routing_cpu != &execution : bool(routing_cpu) != (staged || cpu_layer || policy.cpu_share > 0))
                throw std::runtime_error("MoE scheduler did not lazily create its required CPU context");
            if (cpu && routing_cpu != cpu)
                throw std::runtime_error("MoE scheduler replaced an existing CPU context");
            core::PreparedGraph second_owner(execution, {"test.moe.scheduler.reuse", 1024 * 1024, 128});
            if (bool(resources.configure_scheduler(second_owner)) != (staged || (cpu_layer && !host_primary)) ||
                resources.cpu_execution() != routing_cpu)
                throw std::runtime_error("MoE scheduler did not reuse its routing CPU context");
        }
        if (staged) {
            const auto stats = resources.disk_store()->statistics();
            if (resources.disk_store()->cpu_execution() != cpu ||
                !stats.buffer_bytes || stats.host_buffer_bytes || stats.expert_loads || stats.bytes_read)
                throw std::runtime_error("unpinned MoE staging eagerly materialized expert banks");
        }
        core::PreparedGraph graph(target, {"test.moe.resources", 4 * 1024 * 1024, 256, 16 * ggml_tensor_overhead()});
        auto * meta = graph.persistent_build_context().ggml;
        auto * input = ggml_new_tensor_3d(meta, GGML_TYPE_F32, 256, 1, 2);
        auto * ids = ggml_new_tensor_2d(meta, GGML_TYPE_I32, 2, 2);
        ggml_set_input(input);
        ggml_set_input(ids);
        graph.allocate_persistent();
        auto & ctx = graph.build_context();
        auto * local = staged ? middle.disk_store->route(ctx, ids, middle.disk_group) : ids;
        std::vector<ggml_tensor *> outputs, references;
        for (size_t p = 0; p < names.size(); ++p) {
            auto * bank = middle.weights[p].tensor;
            if (bank->ne[2] != (staged ? 4 : 12) || bank->type != full[p].tensor->type ||
                bool(middle.disk_store) != staged)
                throw std::runtime_error("MoE bank lost native type, residency or cache bound");
            outputs.push_back(ggml_mul_mat_id(ctx.ggml, bank, input, local));
            references.push_back(ggml_mul_mat_id(ctx.ggml, expected_weights[p].tensor, input, ids));
            ggml_set_output(outputs.back());
            ggml_set_output(references.back());
            ggml_build_forward_expand(graph.graph(), outputs.back());
            ggml_build_forward_expand(graph.graph(), references.back());
        }
        if (staged) {
            // Accelerator schedulers need the coordinator's lazy CPU backend
            // last; a CPU-target projection already has that backend locally.
            auto * scheduler = core::is_host_backend(target.backend())
                ? graph.create_scheduler({target.backend()}) : resources.configure_scheduler(graph);
            ggml_backend_sched_set_eval_callback(scheduler, core::DiskExpertWeightStore::evaluate, resources.disk_store());
        }
        graph.allocate();
        std::vector<float> values(512);
        for (size_t i = 0; i < values.size(); ++i) values[i] = 0.01F * std::sin(float(i) * 0.07F);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        const size_t cache_bytes = staged ? resources.disk_store()->statistics().buffer_bytes : 0;
        for (const auto & routes : std::vector<std::vector<int32_t>>{{0, 1, 0, 2}, {8, 9, 10, 11}, {0, 1, 0, 2}}) {
            ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
            const auto slots = staged ? resources.disk_store()->stage(middle.disk_group, routes) : routes;
            for (size_t p = 0; p < names.size(); ++p) {
                for (size_t lane = 0; lane < routes.size(); ++lane) {
                    const size_t bytes = full[p].tensor->nb[2];
                    std::vector<std::byte> actual(bytes), expected(bytes);
                    ggml_backend_tensor_get(middle.weights[p].tensor, actual.data(), size_t(slots[lane]) * bytes, bytes);
                    ggml_backend_tensor_get(full[p].tensor, expected.data(), size_t(routes[lane]) * bytes, bytes);
                    if (actual != expected) throw std::runtime_error("MoE resources changed native expert bytes after source release");
                }
            }
            graph.compute();
            for (size_t p = 0; p < names.size(); ++p) {
                const auto actual = core::read_tensor_f32(outputs[p]);
                const auto expected = core::read_tensor_f32(references[p]);
                if (actual.size() != 1024 || expected.size() != actual.size())
                    throw std::runtime_error("MoE resource projection shape changed");
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::abs(actual[i] - expected[i]) > 1e-4F)
                        throw std::runtime_error("MoE resource projection differs from native resident bank");
                    if (full[p].tensor->type == GGML_TYPE_F32) {
                        const size_t lane = i / 256, row = i % 256, token = lane / 2;
                        double sum = 0;
                        for (size_t col = 0; col < 256; ++col) {
                            float weight;
                            const size_t offset = ((size_t(routes[lane]) * 256 + row) * 256 + col) * sizeof(float);
                            std::memcpy(&weight, entries[p].data.data() + offset, sizeof(float));
                            sum += double(weight) * values[token * 256 + col];
                        }
                        if (std::abs(double(actual[i]) - sum) > 1e-4)
                            throw std::runtime_error("MoE resource projection differs from independent F32 dot product");
                    }
                }
            }
            if (&middle != &resources.layer(1) || (staged && resources.disk_store()->statistics().buffer_bytes != cache_bytes))
                throw std::runtime_error("MoE execution changed a handle or bounded cache allocation");
        }
        if (staged && cpu_layer) {
            bounded_cpu_staged_execution(execution, *cpu, *resources.disk_store(), middle.disk_group, expected_weights);
            // Use the primary backbone context around borrowed-CPU execution,
            // retaining the independent F32 host oracle and every BF16 gate.
            if (full[0].tensor->type == GGML_TYPE_F32)
                dense_hybrid_oracle(execution, *resources.disk_store(), middle.disk_group, entries, true);
        } else if (staged && full[0].tensor->type == GGML_TYPE_F32) {
            dense_hybrid_oracle(target, *resources.disk_store(), middle.disk_group, entries);
        }
        std::cout << "moe_resources type=" << ggml_type_name(full[0].tensor->type)
                  << " staged=" << staged << " cpu_layer=" << cpu_layer << " cycle=" << cycle << '\n';
    }
}


void pinned_cpu_expert_resources(const std::filesystem::path & path, engine::core::ExecutionContext & execution,
    engine::core::ExecutionContext & cpu, const std::vector<std::string> & names,
    const std::vector<engine::io::SafeTensorWriteEntry> & entries) {
    namespace core = engine::core;
    namespace assets = engine::assets;
    const auto provider = execution.backend();
    const bool supported = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(provider)) != nullptr;
    auto source = assets::open_tensor_source(path);
    std::vector<core::MoeExpertTensorBinding> bindings;
    for (const auto & name : names)
        bindings.push_back({name, name, {12, 256, 256}, assets::TensorStorageType::Native});
    core::BackendWeightStore reference(cpu.backend(), cpu.backend_type(), "test.moe.pinned_cpu.reference", 1024 * 1024);
    std::vector<core::TensorValue> resident;
    if (supported) {
        for (const auto & name : names)
            resident.push_back(reference.load_tensor(*source, name, assets::TensorStorageType::Native, {12, 256, 256}));
        reference.upload();
        source->release_storage();
    }
    {
        // CPU owns the cache and executes experts; the primary backend supplies
        // pinned host allocations independently of that execution placement.
        core::DiskExpertWeightStore store(path, true, 1, &cpu, provider);
        bool rejected = false;
        size_t group = 0;
        try { group = store.add_group(names, cpu.backend(), 4); }
        catch (const std::runtime_error & error) {
            if (supported || std::string(error.what()).find("pinned host buffers") == std::string::npos) throw;
            rejected = true;
        }
        if (!supported && !rejected) throw std::runtime_error("CPU cache accepted an unsupported pinning provider");
        if (supported) {
            bounded_cpu_staged_execution(execution, cpu, store, group, resident, true);
            if (resident[0].tensor->type == GGML_TYPE_F32)
                dense_hybrid_oracle(execution, store, group, entries, true, true);
        }
    }
    {
        core::BackendWeightStore primary(execution.backend(), execution.backend_type(), "test.moe.pinned_cpu.primary", 1024 * 1024);
        const std::vector<float> norm_values(256, 1.0F);
        const auto norm = primary.make_f32(core::TensorShape::from_dims({256}), norm_values);
        core::MoeExecutionPolicy policy;
        policy.residency = core::MoeExpertResidency::Staged;
        policy.pin_host_weights = true;
        policy.cache_slots = 4;
        policy.cpu_layers = {0};
        policy.cpu_share = execution.moe_cpu_share();
        core::MoeResources resources(execution, *source, primary, policy, "test.moe.pinned_cpu.weights", 1024 * 1024);
        bool rejected = false;
        try { resources.bind_layer(0, 5, bindings, 2); }
        catch (const std::runtime_error & error) {
            if (supported || std::string(error.what()).find("pinned host buffers") == std::string::npos) throw;
            rejected = true;
        }
        if (!supported && !rejected) throw std::runtime_error("MoE resources accepted an unsupported pinning provider");
        if (supported) {
            primary.upload();
            resources.finalize_weights();
            if (core::read_tensor_f32(norm.tensor) != norm_values || !resources.cpu_execution() ||
                resources.disk_store()->cpu_execution() != resources.cpu_execution())
                throw std::runtime_error("pinned CPU resources changed primary upload or borrowed CPU ownership");
            bounded_cpu_staged_execution(execution, *resources.cpu_execution(), *resources.disk_store(),
                resources.layer(0).disk_group, resident, true);
            if (resident[0].tensor->type == GGML_TYPE_F32)
                dense_hybrid_oracle(execution, *resources.disk_store(), resources.layer(0).disk_group, entries, true, true);
        }
    }
    std::cout << "pinned_cpu_provider supported=" << supported << '\n';
}


void concurrent_moe_sessions(const std::filesystem::path & path, const engine::core::BackendConfig & config,
    const std::vector<std::string> & names, const std::vector<engine::io::SafeTensorWriteEntry> & entries) {
    namespace core = engine::core;
    namespace assets = engine::assets;
    ConcurrentMoeOverlap barrier;
    std::array<std::exception_ptr, 2> errors{};
    std::array<std::thread, 2> sessions;
    const auto run = [&](size_t session) {
        try {
            const bool accelerator = config.type != core::BackendType::Cpu;
            core::ExecutionContext execution(config, accelerator ? 0.5F : 0.0F);
            auto source = assets::open_tensor_source(path);
            core::BackendWeightStore weights(execution.backend(), execution.backend_type(), "test.moe.concurrent.primary", 1024 * 1024);
            const auto norm = weights.make_f32(core::TensorShape::from_dims({256}), std::vector<float>(256, 1.0F));
            core::MoeExecutionPolicy policy;
            policy.residency = core::MoeExpertResidency::Staged;
            policy.cache_slots = 4;
            policy.cpu_share = execution.moe_cpu_share();
            if (!accelerator) policy.cpu_layers = {0};
            core::MoeResources resources(execution, *source, weights, policy, "test.moe.concurrent.cpu", 1024 * 1024);
            std::vector<core::MoeExpertTensorBinding> bindings;
            for (const auto & name : names)
                bindings.push_back({name, name, {12, 256, 256}, assets::TensorStorageType::Native});
            const auto & layer = resources.bind_layer(0, 2 + session, bindings, 2);
            weights.upload();
            resources.finalize_weights();
            source->release_storage();
            auto * cpu = resources.cpu_execution();
            if (!cpu || !core::is_host_backend(cpu->backend()) ||
                resources.disk_store()->cpu_execution() != cpu ||
                core::read_tensor_f32(norm.tensor) != std::vector<float>(256, 1.0F))
                throw std::runtime_error("concurrent session lost its owned weights or CPU borrower");
            {
                std::lock_guard<std::mutex> lock(barrier.mutex);
                if (barrier.cancelled) throw std::runtime_error("concurrent MoE peer failed during preparation");
                const auto peer = 1 - session;
                // Compare captured identities without dereferencing peer owners:
                // a failed peer may already be unwinding its resource lifetime.
                if (barrier.primary[peer] && (barrier.primary[peer] == execution.backend() ||
                    barrier.cpu[peer] == cpu->backend() || barrier.resources[peer] == &resources ||
                    barrier.stores[peer] == resources.disk_store() ||
                    barrier.banks[peer] == layer.weights[0].tensor ||
                    barrier.buffers[peer] == layer.weights[0].tensor->buffer))
                    throw std::runtime_error("independent MoE sessions share a mutable backend or expert cache");
                barrier.primary[session] = execution.backend();
                barrier.cpu[session] = cpu->backend();
                barrier.stores[session] = resources.disk_store();
                barrier.resources[session] = &resources;
                barrier.banks[session] = layer.weights[0].tensor;
                barrier.buffers[session] = layer.weights[0].tensor->buffer;
            }
            // Install the existing GGML callback before graph plans are made;
            // CPU plans retain it, and accelerator hybrid workers execute it.
            using SetAbortCallback = void (*)(ggml_backend_t, ggml_abort_callback, void *);
            const auto registry = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu->backend()));
            const auto set_abort = reinterpret_cast<SetAbortCallback>(
                ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_abort_callback"));
            if (!set_abort) throw std::runtime_error("CPU backend does not expose its execution callback");
            ConcurrentMoeOverlap::Participant participant{barrier, session};
            set_abort(cpu->backend(), ConcurrentMoeOverlap::expert_callback, &participant);
            const auto before = resources.disk_store()->statistics();
            try {
                dense_hybrid_oracle(execution, *resources.disk_store(), layer.disk_group, entries, !accelerator, false, &participant);
            } catch (...) {
                set_abort(cpu->backend(), nullptr, nullptr);
                throw;
            }
            set_abort(cpu->backend(), nullptr, nullptr);
            const auto after = resources.disk_store()->statistics();
            if (after.buffer_bytes != before.buffer_bytes || (!accelerator && after.host_buffer_bytes != 0) ||
                &resources.layer(0) != &layer)
                throw std::runtime_error("concurrent execution grew expert caches or invalidated a session handle");
        } catch (...) {
            errors[session] = std::current_exception();
            barrier.cancel();
        }
    };
    // Thread creation can fail after the first participant starts. Release its
    // rendezvous before joining; never leave a peer blocked on fixture failure.
    try {
        for (size_t session = 0; session < sessions.size(); ++session)
            sessions[session] = std::thread(run, session);
    } catch (...) {
        barrier.cancel();
        for (auto & session : sessions) if (session.joinable()) session.join();
        throw;
    }
    for (auto & session : sessions) session.join();
    for (const auto & error : errors) if (error) std::rethrow_exception(error);
    if (barrier.cancelled || barrier.waiting != 0 || barrier.generation != 16)
        throw std::runtime_error("independent MoE sessions did not overlap all expert executions");
    std::cout << "concurrent_moe_sessions sessions=2 expert_rendezvous=" << barrier.generation
              << " hybrid=" << (config.type != core::BackendType::Cpu) << '\n';
}

void moe_resource_rejections(const std::filesystem::path & path, engine::core::ExecutionContext & requested_execution,
    const std::vector<std::string> & names) {
    namespace core = engine::core;
    namespace assets = engine::assets;
    core::ExecutionContext execution(requested_execution.config());
    for (const auto & shares : {std::pair<float, float>{0, 0.5F}, {0.5F, 0}, {0.5F, 0.75F}}) {
        core::ExecutionContext context(execution.config(), shares.first);
        auto tensors = assets::open_tensor_source(path);
        core::BackendWeightStore weights(context.backend(), context.backend_type(), "test.moe.mismatch", 1024 * 1024);
        core::MoeExecutionPolicy policy;
        policy.residency = core::MoeExpertResidency::Staged;
        policy.cpu_share = shares.second;
        bool rejected = false;
        try { core::MoeResources resources(context, *tensors, weights, policy, "test.moe.cpu", 1024 * 1024); }
        catch (const std::invalid_argument & error) {
            rejected = std::string(error.what()) == "MoE policy CPU share must match the execution context";
        }
        if (!rejected) throw std::runtime_error("mismatched MoE CPU share accepted");
    }
    auto source = assets::open_tensor_source(path);
    std::vector<core::MoeExpertTensorBinding> bindings;
    for (const auto & name : names)
        bindings.push_back({name, name, {12, 256, 256}, assets::TensorStorageType::Native});
    for (int invalid = 0; invalid < 8; ++invalid) {
        core::MoeExecutionPolicy bad;
        if (invalid == 0) bad.cache_slots = 0;
        if (invalid == 1) bad.read_threads = 0;
        if (invalid == 2) bad.read_threads = 65;
        if (invalid == 3) bad.cpu_share = -0.1F;
        if (invalid == 4) bad.cpu_share = 1.1F;
        if (invalid == 5) bad.cpu_share = std::numeric_limits<float>::quiet_NaN();
        if (invalid == 6) bad.cpu_share = 0.5F; // Resident hybrid is unsupported.
        if (invalid == 7) bad.cpu_layers = {0, 0};
        core::BackendWeightStore weights(execution.backend(), execution.backend_type(), "test.moe.policy", 1024 * 1024);
        bool rejected = false;
        try { core::MoeResources resources(execution, *source, weights, bad, "test.moe.policy.cpu", 1024 * 1024); }
        catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid direct MoE resource policy accepted");
    }
    // Every failed bind must leave ordinal zero available for a valid retry.
    for (int invalid = 0; invalid < 10; ++invalid) {
        core::BackendWeightStore weights(execution.backend(), execution.backend_type(), "test.moe.invalid", 1024 * 1024);
        core::MoeExecutionPolicy policy;
        policy.residency = core::MoeExpertResidency::Staged;
        policy.cache_slots = invalid == 6 ? 1 : invalid == 7 ? 13 : 4;
        core::MoeResources resources(execution, *source, weights, policy, "test.moe.invalid.cpu", 1024 * 1024);
        auto bad = bindings;
        if (invalid == 1) bad.clear();
        if (invalid == 3) bad[0].shape = {12, 256};
        if (invalid == 4) bad[1].shape = {11, 256, 256};
        if (invalid == 8) bad[0].shape = {12, 255, 256};
        if (invalid == 5) bad[0].storage = assets::TensorStorageType::F32;
        bool rejected = false;
        try { resources.bind_layer(invalid == 0 ? 1 : 0, 7, bad, invalid == 2 ? 0 : invalid == 9 ? 13 : 2); }
        catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid MoE binding/cache accepted");
        if (invalid != 6 && invalid != 7) {
            resources.bind_layer(0, 7, bindings, 2);
            // Staged experts do not populate the caller's primary store. A real
            // session still uploads its backbone norm before finalizing caches.
            const std::vector<float> norm_values(256, 1.0F);
            const auto primary_norm = weights.make_f32(core::TensorShape::from_dims({256}), norm_values);
            weights.upload();
            if (core::read_tensor_f32(primary_norm.tensor) != norm_values)
                throw std::runtime_error("MoE rejection retry did not upload primary backbone weights");
            resources.finalize_weights();
        }
    }
    core::BackendWeightStore weights(execution.backend(), execution.backend_type(), "test.moe.finalize", 1024 * 1024);
    core::MoeExecutionPolicy policy;
    policy.cpu_layers = {1};
    core::MoeResources resources(execution, *source, weights, policy, "test.moe.finalize.cpu", 1024 * 1024);
    const auto & first = resources.bind_layer(0, 2, bindings, 2);
    bool rejected = false;
    try { resources.bind_layer(0, 3, bindings, 2); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("duplicate MoE ordinal accepted");
    rejected = false;
    try { resources.finalize_weights(); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("MoE finalized an unregistered CPU ordinal");
    rejected = false;
    try { resources.layer(1); }
    catch (const std::out_of_range &) { rejected = true; }
    if (!rejected) throw std::runtime_error("unregistered MoE handle lookup accepted");
    resources.bind_layer(1, 9, bindings, 2);
    weights.upload();
    resources.finalize_weights();
    if (&first != &resources.layer(0)) throw std::runtime_error("finalization retry invalidated a MoE handle");
    rejected = false;
    try { resources.finalize_weights(); }
    catch (const std::logic_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("duplicate MoE finalization accepted");
    rejected = false;
    try { resources.bind_layer(2, 12, bindings, 2); }
    catch (const std::logic_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("MoE bind after finalization accepted");
}

}

int main(int argc, char ** argv) {
    try {
        namespace core = engine::core;
        namespace assets = engine::assets;
        engine::debug::configure_logging({true, std::nullopt});
        const auto started = std::chrono::steady_clock::now();
        core::BackendConfig config;
        config.threads = 8;
        if (argc > 1 && std::string(argv[1]) == "cuda") config.type = core::BackendType::Cuda;
        if (argc > 1 && std::string(argv[1]) == "vulkan") config.type = core::BackendType::Vulkan;
        if (argc > 1 && std::string(argv[1]) == "metal") config.type = core::BackendType::Metal;
        float moe_cpu_share = 0;
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == "--moe-cpu-share") moe_cpu_share = std::stof(argv[i + 1]);
        core::ExecutionContext execution(config, moe_cpu_share);
        std::unique_ptr<core::ExecutionContext> host;
        if (config.type != core::BackendType::Cpu) {
            auto host_config = config;
            host_config.type = core::BackendType::Cpu;
            host = std::make_unique<core::ExecutionContext>(host_config);
        }
        auto * cpu_execution = host ? host.get() : &execution;
        moe_policy_resolution();
        FixtureDirectory directory;
        const auto input = directory.path / "weights.safetensors";
        const auto gguf = directory.path / "weights.gguf";
        const std::vector<std::string> names = {
            "model.layers.0.mlp.experts.gate.weight", "model.layers.0.mlp.experts.up.weight",
            "model.layers.0.mlp.experts.down.weight"};
        std::vector<engine::io::SafeTensorWriteEntry> entries;
        for (size_t p = 0; p < names.size(); ++p) {
            std::vector<float> values(12 * 256 * 256);
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * .013F + float(p)) * .1F;
            std::vector<unsigned char> bytes(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            entries.push_back({names[p], "F32", {12, 256, 256}, std::move(bytes)});
        }
        engine::io::write_safetensors_file(input, entries);
        for (const size_t invalid_threads : {size_t{0}, size_t{65}}) {
            bool rejected = false;
            try { core::DiskExpertWeightStore invalid(gguf, false, invalid_threads, cpu_execution); }
            catch (const std::invalid_argument &) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid expert reader count was not rejected");
        }
        const std::vector<std::tuple<assets::TensorStorageType, bool, size_t>> modes{
            {assets::TensorStorageType::F32, false, 1}, {assets::TensorStorageType::Q4_K, false, 1},
            {assets::TensorStorageType::F32, true, 1}, {assets::TensorStorageType::Q4_K, true, 1},
#if defined(_OPENMP) || defined(__APPLE__)
            {assets::TensorStorageType::F32, false, 4}, {assets::TensorStorageType::Q4_K, false, 4},
            {assets::TensorStorageType::F32, false, 64}, {assets::TensorStorageType::Q4_K, false, 64},
#endif
        };
        for (const auto & [type, pin_experts, read_threads] : modes) {
            assets::GgufConversionOptions options;
            options.type_overrides.push_back({"model.layers.0.mlp.experts.*", type});
            assets::convert_tensor_sources_to_gguf({{input, ""}}, gguf, assets::TensorStorageType::F32,
                true, false, {}, {}, std::nullopt, options);
            core::DiskExpertWeightStore disk(gguf, pin_experts, read_threads, cpu_execution);
            if (disk.cpu_execution() != cpu_execution)
                throw std::runtime_error("disk expert store did not retain its borrowed CPU context");
#if !defined(_OPENMP) && !defined(__APPLE__)
            bool parallel_rejected = false;
            try { core::DiskExpertWeightStore parallel(gguf, false, 4, cpu_execution); }
            catch (const std::runtime_error & error) {
                parallel_rejected = std::string(error.what()).find("ENGINE_ENABLE_OPENMP") != std::string::npos;
            }
            if (!parallel_rejected) throw std::runtime_error("parallel reads without OpenMP were not rejected");
#endif
            if (pin_experts && !ggml_backend_dev_host_buffer_type(ggml_backend_get_device(execution.backend()))) {
                bool rejected = false;
                try { disk.add_group(names, execution.backend(), 4); }
                catch (const std::runtime_error & error) {
                    rejected = std::string(error.what()).find("pinned host buffers") != std::string::npos;
                }
                if (!rejected) throw std::runtime_error("unsupported pinned experts were not rejected");
                continue;
            }
            const auto group = disk.add_group(names, execution.backend(), 4);
            if (type == assets::TensorStorageType::F32 && read_threads == 1)
                check_group_registration_failures(gguf, execution, names, pin_experts, cpu_execution);
            engine::modules::MoePlacementPlan placement_plan(12);
            for (const float share : {-0.1F, 1.1F, std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::quiet_NaN()}) {
                bool rejected = false;
                try { placement_plan.update({0}, disk.cached_experts(group), disk.cache_epoch(), share); }
                catch (const std::invalid_argument &) { rejected = true; }
                if (!rejected) throw std::runtime_error("invalid MoE CPU share was accepted");
            }
            placement_plan.update({0, 1, 0}, disk.cached_experts(group), disk.cache_epoch(), 1);
            const auto & cold = placement_plan;
            if (cold.gpu != std::vector<int32_t>({0, 1}) || !cold.cpu.empty())
                throw std::runtime_error("hybrid placement did not fill the cold cache first");
            if ((!pin_experts && disk.statistics().bytes_read != 0) || disk.statistics().expert_loads != 0)
                throw std::runtime_error("expert payloads were eagerly read");
            if (pin_experts && disk.statistics().host_buffer_bytes == 0)
                throw std::runtime_error("pinned expert banks were not allocated");
            auto source = assets::open_tensor_source(gguf);
            const auto expected_type = type == assets::TensorStorageType::Q4_K ? "q4_k" : "f32";
            if (source->require_metadata(names[0]).dtype != expected_type)
                throw std::runtime_error("fixture expert dtype mismatch");
            core::BackendWeightStore resident(execution.backend(), execution.backend_type(), "test.resident", 1024 * 1024);
            std::vector<core::TensorValue> full;
            for (const auto & name : names) full.push_back(resident.load_tensor(*source, name,
                assets::TensorStorageType::Native, {12, 256, 256}));
            resident.upload();
            if (!pin_experts && read_threads == 1) {
                moe_resource_rejections(gguf, execution, names);
                moe_resource_matrix(gguf, execution, names, full, entries);
                pinned_cpu_expert_resources(gguf, execution, *cpu_execution, names, entries);
                if (type == assets::TensorStorageType::F32) concurrent_moe_sessions(gguf, config, names, entries);
            }
            if (read_threads == 1) check_host_bank_failures(gguf, execution, names, full, pin_experts, cpu_execution);
            std::unique_ptr<ggml_context, ContextDeleter> inputs(ggml_init({1024 * 1024, nullptr, true}));
            std::unique_ptr<ggml_context, ContextDeleter> context(ggml_init({4 * 1024 * 1024, nullptr, true}));
            auto * ids = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, 2, 2);
            auto * x = ggml_new_tensor_3d(inputs.get(), GGML_TYPE_F32, 256, 1, 2);
            ggml_set_input(ids);
            ggml_set_input(x);
            std::unique_ptr<ggml_backend_buffer, BufferDeleter> input_buffer(
                ggml_backend_alloc_ctx_tensors(inputs.get(), execution.backend()));
            core::ModuleBuildContext ctx{context.get(), "disk_expert_test", execution.backend_type()};
            auto * local_ids = disk.route(ctx, ids, group);
            ggml_tensor * cached = nullptr;
            ggml_tensor * expected = nullptr;
            for (size_t p = 0; p < names.size(); ++p) {
                auto * a = ggml_mul_mat_id(context.get(), disk.tensor(group, p).tensor, x, local_ids);
                auto * b = ggml_mul_mat_id(context.get(), full[p].tensor, x, ids);
                cached = cached ? ggml_add(context.get(), cached, a) : a;
                expected = expected ? ggml_add(context.get(), expected, b) : b;
            }
            auto * graph = ggml_new_graph_custom(context.get(), 128, false);
            ggml_set_output(cached);
            ggml_set_output(expected);
            ggml_build_forward_expand(graph, cached);
            ggml_build_forward_expand(graph, expected);
            ggml_backend_t backends[] = {execution.backend(), host ? host->backend() : nullptr};
            std::unique_ptr<ggml_backend_sched, SchedulerDeleter> scheduler(
                ggml_backend_sched_new(backends, nullptr, host ? 2 : 1, 128, false, false));
            ggml_backend_sched_set_eval_callback(scheduler.get(), core::DiskExpertWeightStore::evaluate, &disk);
            if (!ggml_backend_sched_alloc_graph(scheduler.get(), graph)) throw std::runtime_error("test graph allocation failed");
            std::vector<float> values(512);
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::cos(float(i) * .07F);
            ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
            const auto cache_bytes = disk.statistics().buffer_bytes;
            size_t full_bytes = 0;
            for (const auto & weight : full) full_bytes += ggml_nbytes(weight.tensor);
            if (cache_bytes > full_bytes / 3 + 1024) throw std::runtime_error("expert cache allocation is not bounded by slots");
            int requests = 0;
            for (const auto & route : std::vector<std::vector<int32_t>>{
                     {0, 1, 2, 3}, {0, 1, 2, 3}, {4, 5, 4, 5}, {0, 1, 2, 3}, {0, 1, 2, 3}}) {
                const auto before = disk.statistics();
                ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
                if (ggml_backend_sched_graph_compute(scheduler.get(), graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("expert graph compute failed");
                const auto actual = core::read_tensor_f32(cached);
                const auto reference = core::read_tensor_f32(expected);
                float maximum = 0;
                for (size_t i = 0; i < actual.size(); ++i) maximum = std::max(maximum, std::abs(actual[i] - reference[i]));
                if (!std::all_of(actual.begin(), actual.end(), [](float a) { return std::isfinite(a); }) || maximum > 1e-4F)
                    throw std::runtime_error("disk and resident expert output mismatch: " + std::to_string(maximum));
                const auto after = disk.statistics();
                if ((pin_experts || requests == 1 || requests == 4) && before.bytes_read != after.bytes_read)
                    throw std::runtime_error("cache hit read expert payloads again");
                if (after.buffer_bytes != cache_bytes) throw std::runtime_error("expert cache grew");
                if (core::read_tensor_i32(ids) != route) throw std::runtime_error("routing changed original expert IDs");
                std::cout << "read_threads=" << read_threads << " pin_experts=" << pin_experts << " request=" << requests++ << " max_abs=" << maximum << " expert_loads=" << after.expert_loads
                          << " bytes_read=" << after.bytes_read << " cache_bytes=" << after.buffer_bytes << '\n';
            }
            const std::vector<int32_t> grouped_ids{7, 6, 7, 4};
            const auto slots = disk.stage(group, grouped_ids);
            if (slots.size() != grouped_ids.size() || slots[0] != slots[2])
                throw std::runtime_error("explicit expert staging lost ID order or duplicate IDs");
            for (size_t p = 0; p < full.size(); ++p) {
                const auto bytes = full[p].tensor->nb[2];
                std::vector<std::byte> cached_bytes(bytes), expected_bytes(bytes);
                for (size_t row = 0; row < slots.size(); ++row) {
                    ggml_backend_tensor_get(disk.tensor(group, p).tensor, cached_bytes.data(), slots[row] * bytes, bytes);
                    ggml_backend_tensor_get(full[p].tensor, expected_bytes.data(), grouped_ids[row] * bytes, bytes);
                    if (cached_bytes != expected_bytes) throw std::runtime_error("explicit staging changed native expert bytes");
                }
            }
            const auto grouped_before = disk.statistics();
            if (disk.stage(group, grouped_ids) != slots || disk.statistics().expert_loads != grouped_before.expert_loads)
                throw std::runtime_error("explicit expert staging did not reuse cached slots");
            const auto other_group = disk.add_group(names, execution.backend(), 4);
            disk.stage(other_group, {0, 1, 2, 3});
            placement_plan.update({0, 4, 4, 5}, disk.cached_experts(other_group), disk.cache_epoch(), 0.5F);
            const auto & placement = placement_plan;
            if (placement.gpu != std::vector<int32_t>({0, 4}) || placement.cpu != std::vector<int32_t>({5}))
                throw std::runtime_error("hybrid placement changed hits or counted duplicate misses");
            size_t cpu_misses = 0;
            for (int repeat = 0; repeat < 10; ++repeat) {
                placement_plan.update({4}, disk.cached_experts(other_group), disk.cache_epoch(), 0.5F);
                cpu_misses += placement_plan.cpu.size();
            }
            placement_plan.update({0, 1, 2, 3}, disk.cached_experts(other_group), disk.cache_epoch(), 1);
            if (cpu_misses != 5 || !placement_plan.cpu.empty())
                throw std::runtime_error("hybrid placement lost fractional carry or moved a cache hit to CPU");
            // A miss appears before every hit in the logical order. Executing
            // that order would evict expert 11 and upload it again later.
            disk.invalidate_cache();
            disk.stage(other_group, {8, 9, 10, 11});
            placement_plan.update({0, 8, 9, 10, 11, 0}, disk.cached_experts(other_group), disk.cache_epoch(), 0.5F);
            if (placement_plan.gpu != std::vector<int32_t>({8, 9, 10, 11, 0}) || !placement_plan.cpu.empty())
                throw std::runtime_error("GPU placement failed to consume resident hits first");
            const auto before_mixed = disk.statistics().expert_loads;
            for (size_t begin = 0; begin < placement_plan.gpu.size(); begin += 4) {
                const size_t end = std::min(placement_plan.gpu.size(), begin + size_t{4});
                disk.stage(other_group, {placement_plan.gpu.begin() + begin, placement_plan.gpu.begin() + end});
            }
            if (disk.statistics().expert_loads - before_mixed != 1)
                throw std::runtime_error("GPU execution reloaded a hit from its original placement plan");
            for (const auto & route : std::vector<std::vector<int32_t>>{{7, 6, 7, 4}, {4, 6, 4, 7}, {7}, {7, 6, 7, 4}}) {
                core::DiskExpertWeightStore cold(gguf, pin_experts, read_threads, cpu_execution);
                const auto cold_group = cold.add_group(names, execution.backend(), 4);
                const auto cold_slots = cold.stage(cold_group, route);
                for (const auto cached_group : {group, other_group}) {
                    disk.stage(cached_group, {0, 1, 2, 3});
                    for (const auto id : {7, 6, 4}) disk.stage(cached_group, {id});
                }
                const auto before_reset = disk.statistics();
                disk.invalidate_cache();
                if (disk.statistics().buffer_bytes != before_reset.buffer_bytes ||
                    disk.statistics().host_buffer_bytes != before_reset.host_buffer_bytes ||
                    disk.statistics().expert_loads != before_reset.expert_loads)
                    throw std::runtime_error("invalidating expert placement changed allocations or statistics");
                const auto restaged = disk.stage(group, route);
                if (restaged != cold_slots || disk.stage(other_group, route) != cold_slots ||
                    disk.statistics().expert_loads - before_reset.expert_loads != 2 * cold.statistics().expert_loads)
                    throw std::runtime_error("invalidating experts did not restore empty-cache placement");
                for (size_t row = 0; row < route.size(); ++row) {
                    for (size_t p = 0; p < full.size(); ++p) {
                        const auto bytes = full[p].tensor->nb[2];
                        std::vector<std::byte> cached_bytes(bytes), expected_bytes(bytes);
                        ggml_backend_tensor_get(disk.tensor(group, p).tensor, cached_bytes.data(), restaged[row] * bytes, bytes);
                        ggml_backend_tensor_get(full[p].tensor, expected_bytes.data(), route[row] * bytes, bytes);
                        if (cached_bytes != expected_bytes) throw std::runtime_error("invalidating experts changed staged bytes");
                    }
                }
                const auto loads = disk.statistics().expert_loads;
                if (disk.stage(group, route) != restaged || disk.stage(other_group, route) != restaged ||
                    disk.statistics().expert_loads != loads)
                    throw std::runtime_error("expert staging did not reuse within-request slots");
            }
            bool capacity_rejected = false;
            try { disk.stage(group, {0, 1, 2, 3, 4}); }
            catch (const std::runtime_error & error) {
                capacity_rejected = std::string(error.what()).find("cache capacity") != std::string::npos;
            }
            if (!capacity_rejected) throw std::runtime_error("explicit staging exceeded bounded cache capacity");
            const std::vector<int32_t> invalid{12, 0, 1, 2};
            ggml_backend_tensor_set(ids, invalid.data(), 0, invalid.size() * sizeof(int32_t));
            bool rejected = false;
            try { ggml_backend_sched_graph_compute(scheduler.get(), graph); }
            catch (const std::runtime_error & e) { rejected = std::string(e.what()).find("routing ID out of range") != std::string::npos; }
            if (!rejected) throw std::runtime_error("invalid routing IDs were not rejected");
            if (!pin_experts && read_threads == 1) {
                if (host && type == assets::TensorStorageType::F32) dense_hybrid_oracle(execution, disk, group, entries);
                if (host) automatic_hybrid_graph(execution, *host, disk, group, full);
                if (host && type == assets::TensorStorageType::Q4_K)
                    automatic_hybrid_lifecycle(execution, *host, disk, group, full);
                if (host && execution.moe_cpu_share() == 0.5F && type == assets::TensorStorageType::Q4_K) native_hybrid_lanes(execution, *host, disk, group, full);
                namespace modules = engine::modules;
                // A diagonal expert bank gives an independent host oracle for
                // routing/staging without sharing the implementation under test.
                // Dense arbitrary-weight storage/projection checks run above.
                const auto staged_input = directory.path / "staged.safetensors";
                const auto staged_gguf = directory.path / "staged.gguf";
                std::vector<engine::io::SafeTensorWriteEntry> staged_entries;
                for (size_t p = 0; p < names.size(); ++p) {
                    std::vector<float> values(12 * 256 * 256, 0.0F);
                    for (size_t e = 0; e < 12; ++e) for (size_t row = 0; row < 256; ++row)
                        values[(e * 256 + row) * 256 + row] = float((1 + e) * (1 + (row + p) % 2)) / 32.0F;
                    std::vector<unsigned char> bytes(values.size() * sizeof(float));
                    std::memcpy(bytes.data(), values.data(), bytes.size());
                    staged_entries.push_back({names[p], "F32", {12, 256, 256}, std::move(bytes)});
                }
                engine::io::write_safetensors_file(staged_input, staged_entries);
                assets::convert_tensor_sources_to_gguf({{staged_input, ""}}, staged_gguf, assets::TensorStorageType::F32,
                    true, false, {}, {}, std::nullopt, options);
                core::DiskExpertWeightStore staged_store(staged_gguf, false, 1, cpu_execution);
                const auto staged_group = staged_store.add_group(names, execution.backend(), 4);
                auto staged_source = assets::open_tensor_source(staged_gguf);
                core::BackendWeightStore oracle_store(execution.backend(), execution.backend_type(), "test.oracle", 1024 * 1024);
                std::vector<core::TensorValue> oracle_weights;
                for (const auto & name : names) oracle_weights.push_back(oracle_store.load_tensor(*staged_source, name,
                    assets::TensorStorageType::Native, {12, 256, 256}));
                oracle_store.upload();
                std::vector<std::vector<float>> diagonal(3, std::vector<float>(12 * 256));
                for (size_t p = 0; p < oracle_weights.size(); ++p) {
                    auto * tensor = oracle_weights[p].tensor;
                    std::vector<std::byte> bytes(ggml_nbytes(tensor));
                    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
                    std::vector<float> values(12 * 256 * 256);
                    if (tensor->type == GGML_TYPE_F32) std::memcpy(values.data(), bytes.data(), bytes.size());
                    else ggml_get_type_traits(tensor->type)->to_float(bytes.data(), values.data(), static_cast<int64_t>(values.size()));
                    for (size_t e = 0; e < 12; ++e) for (size_t row = 0; row < 256; ++row) {
                        diagonal[p][e * 256 + row] = values[(e * 256 + row) * 256 + row];
                        for (size_t col = 0; col < 256; ++col)
                            if (col != row && values[(e * 256 + row) * 256 + col] != 0.0F)
                                throw std::runtime_error("staged expert oracle requires diagonal native weights");
                    }
                }
                // Includes every variable row bucket, cache eviction, multiple
                // 32-row blocks and partial tails; repeat after changing routes.
                auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
                const auto shares = std::vector<float>{execution.moe_cpu_share()};
                for (auto share : shares) for (bool relu : {false, true}) for (int tokens : {1, 7, 19, 31, 32, 33, 37, 65, 73}) {
                    const bool drama_boundary = !relu && (tokens == 31 || tokens == 32 || tokens == 33 || tokens == 65);
                    if (relu && (tokens == 31 || tokens == 32 || tokens == 33 || tokens == 65)) continue;
                    workspace->reset();
                    staged_store.stage(staged_group, {8, 9, 10, 11});
                    modules::StagedMoeConfig staged_config;
                    staged_config.hidden_size = 256;
                    staged_config.tokens = tokens;
                    staged_config.selected = 2;
                    staged_config.experts = 12;
                    staged_config.cache_slots = 4;
                    staged_config.rows_per_expert = relu ? std::vector<int64_t>{1, 2, 4, 8, 16, 32} : std::vector<int64_t>{32};
                    staged_config.scatter_index_type = relu ? GGML_TYPE_I64 : GGML_TYPE_I32;
                    modules::StagedMoeRuntime staged(execution, staged_config, workspace, cpu_execution);
                    modules::MoeModule module({relu ? modules::MoeActivation::ReluSquared : modules::MoeActivation::SwiGlu});
                    const auto staged_layer = staged.add_layer(module,
                        {staged_store.tensor(staged_group, 1).tensor, relu ? nullptr : staged_store.tensor(staged_group, 0).tensor,
                         staged_store.tensor(staged_group, 2).tensor}, staged_store, staged_group);
                    std::unique_ptr<ggml_context, ContextDeleter> test_inputs(ggml_init({1024 * 1024, nullptr, true}));
                    std::unique_ptr<ggml_context, ContextDeleter> test_context(ggml_init({4 * 1024 * 1024, nullptr, true}));
                    auto * hidden = ggml_new_tensor_2d(test_inputs.get(), GGML_TYPE_F32, 256, tokens);
                    auto * logical = ggml_new_tensor_2d(test_inputs.get(), GGML_TYPE_I32, 2, tokens);
                    auto * mixing = ggml_new_tensor_2d(test_inputs.get(), GGML_TYPE_F32, 2, tokens);
                    for (auto * tensor : {hidden, logical, mixing}) ggml_set_input(tensor);
                    std::unique_ptr<ggml_backend_buffer, BufferDeleter> test_buffer(ggml_backend_alloc_ctx_tensors(test_inputs.get(), execution.backend()));
                    core::ModuleBuildContext test_ctx{test_context.get(), "test.staged_moe", execution.backend_type()};
                    test_ctx.elide_identity_reshapes = relu;
                    auto * route_graph = ggml_new_graph_custom(test_ctx.ggml, 128, false);
                    staged.build_inputs(test_ctx, route_graph, hidden, {logical, mixing});
                    auto * finish_graph = ggml_new_graph_custom(test_ctx.ggml, 128, false);
                    auto * result = modules::MoeModule::combine(test_ctx, staged.output(), staged.routing_weights());
                    ggml_build_forward_expand(finish_graph, ggml_cpy(test_ctx.ggml, result, staged.hidden_input()));
                    workspace->add(route_graph);
                    workspace->add(finish_graph);
                    auto tail_config = staged_config;
                    tail_config.tokens = std::max<int64_t>(1, tokens - 1);
                    modules::StagedMoeRuntime tail(execution, tail_config, workspace, cpu_execution);
                    const auto tail_layer = tail.add_layer(module,
                        {staged_store.tensor(staged_group, 1).tensor, relu ? nullptr : staged_store.tensor(staged_group, 0).tensor,
                         staged_store.tensor(staged_group, 2).tensor}, staged_store, staged_group);
                    auto * tail_route = ggml_new_graph_custom(test_ctx.ggml, 128, false);
                    tail.build_inputs(test_ctx, tail_route,
                        ggml_view_2d(test_ctx.ggml, hidden, 256, tail_config.tokens, hidden->nb[1], 0),
                        {ggml_view_2d(test_ctx.ggml, logical, 2, tail_config.tokens, logical->nb[1], 0),
                         ggml_view_2d(test_ctx.ggml, mixing, 2, tail_config.tokens, mixing->nb[1], 0)});
                    auto * tail_finish = ggml_new_graph_custom(test_ctx.ggml, 128, false);
                    ggml_build_forward_expand(tail_finish, ggml_cpy(test_ctx.ggml,
                        modules::MoeModule::combine(test_ctx, tail.output(), tail.routing_weights()), tail.hidden_input()));
                    workspace->add(tail_route);
                    workspace->add(tail_finish);
                    // Both graph owners register before the single shared allocation.
                    workspace->prepare();
                    staged.prepare();
                    tail.prepare();
                    core::HostGraphPlan tail_route_plan, tail_finish_plan;
                    core::prepare_host_graph_plan(execution, tail_route, tail_route_plan);
                    core::prepare_host_graph_plan(execution, tail_finish, tail_finish_plan);
                    core::HostGraphPlan route_plan, finish_plan;
                    core::prepare_host_graph_plan(execution, route_graph, route_plan);
                    core::prepare_host_graph_plan(execution, finish_graph, finish_plan);
                    std::vector<float> hidden_values(256 * tokens), mixing_values(2 * tokens);
                    std::vector<int32_t> routes(2 * tokens);
                    for (size_t i = 0; i < hidden_values.size(); ++i) hidden_values[i] = .1F * std::sin(float(i) * .017F);
                    for (size_t i = 0; i < mixing_values.size(); ++i) mixing_values[i] = i % 2 ? .7F : .3F;
                    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
                    ggml_backend_tensor_set(mixing, mixing_values.data(), 0, mixing_values.size() * sizeof(float));
                    const auto workspace_bytes = staged.workspace_bytes();
                    for (int request = 0; request < 4; ++request) {
                        for (size_t i = 0; i < hidden_values.size(); ++i)
                            hidden_values[i] = .1F * std::sin(float(i) * .017F + float(request));
                        ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
                        const bool concentrated = request == 2;
                        for (int token = 0; token < tokens; ++token) {
                            routes[2 * token] = concentrated ? 0 : request == 1 ? (token + 5) % 12 : token % 5;
                            routes[2 * token + 1] = concentrated ? 1 : request == 1 ? (token + 6) % 12 : (token + 1) % 5;
                        }
                        ggml_backend_tensor_set(logical, routes.data(), 0, routes.size() * sizeof(int32_t));
                        if (core::compute_graph(execution, route_graph, route_plan, "test.staged_moe.inputs") != GGML_STATUS_SUCCESS)
                            throw std::runtime_error("staged MoE input execution failed");
                        const auto work_rows = staged.run(staged_layer);
                        if (drama_boundary && concentrated) {
                            // Drama3 uses unrounded SwiGLU, I32 scatter and a
                            // fixed 32-row bucket. Each selected expert receives
                            // exactly tokens assignments, independently of top-k.
                            for (int expert : {0, 1})
                                if (std::count(routes.begin(), routes.end(), expert) != tokens)
                                    throw std::runtime_error("Drama3 boundary assignment count mismatch");
                            const auto expected_rows = uint64_t((tokens + 31) / 32 * 32 * staged_config.cache_slots);
                            if (share == 0 && work_rows != expected_rows)
                                throw std::runtime_error("Drama3 boundary did not execute expected padded batches");
                            std::cout << "drama_boundary assignments_per_expert=" << tokens
                                      << " work_rows=" << work_rows << " cpu_share=" << share << '\n';
                        }
                        if (core::compute_graph(execution, finish_graph, finish_plan, "test.staged_moe.finish") != GGML_STATUS_SUCCESS)
                            throw std::runtime_error("staged MoE output execution failed");
                        if (core::compute_graph(execution, tail_route, tail_route_plan, "test.staged_moe.tail_inputs") != GGML_STATUS_SUCCESS)
                            throw std::runtime_error("staged MoE tail input execution failed");
                        tail.run(tail_layer);
                        if (core::compute_graph(execution, tail_finish, tail_finish_plan, "test.staged_moe.tail_finish") != GGML_STATUS_SUCCESS)
                            throw std::runtime_error("staged MoE tail output execution failed");
                        double maximum = 0;
                        for (const auto & actual : {core::read_tensor_f32(staged.hidden_input()), core::read_tensor_f32(tail.hidden_input())})
                        for (size_t i = 0; i < actual.size(); ++i) {
                            if (!std::isfinite(actual[i])) throw std::runtime_error("staged MoE produced nonfinite output");
                            const size_t token = i / 256, channel = i % 256;
                            double expected = 0;
                            for (size_t lane = 0; lane < 2; ++lane) {
                                const size_t expert = static_cast<size_t>(routes[2 * token + lane]);
                                const size_t index = expert * 256 + channel;
                                const double up = double(hidden_values[i]) * diagonal[1][index];
                                const double gate = double(hidden_values[i]) * diagonal[0][index];
                                const double activated = relu ? std::max(up, 0.0) * std::max(up, 0.0) : gate / (1.0 + std::exp(-gate)) * up;
                                expected += activated * diagonal[2][index] * mixing_values[2 * token + lane];
                            }
                            maximum = std::max(maximum, std::abs(double(actual[i]) - expected));
                        }
                        if (maximum > 1e-4 || core::read_tensor_i32(logical) != routes || workspace_bytes != staged.workspace_bytes())
                            throw std::runtime_error("staged MoE changed assignments, expert output or allocation: " + std::to_string(maximum));
                        std::cout << "staged type=" << expected_type << " cpu_share=" << share << " relu=" << relu << " tokens=" << tokens
                                  << " request=" << request << " max_abs=" << maximum << '\n';
                    }
                    const bool enabled = share > 0 && !core::is_host_backend(execution.backend());
                    if (enabled && staged.statistics().cpu_assignments == 0)
                        throw std::runtime_error("hybrid test never executed a CPU expert");
                    if (!enabled && staged.statistics().cpu_assignments != 0)
                        throw std::runtime_error("CPU/default execution unexpectedly used the hybrid worker");
                    core::release_backend_graph_resources(execution.backend(), route_graph, true);
                    core::release_backend_graph_resources(execution.backend(), finish_graph, true);
                    core::release_backend_graph_resources(execution.backend(), tail_route, true);
                    core::release_backend_graph_resources(execution.backend(), tail_finish, true);
                }
            }
            source->release_storage();
#if !defined(_WIN32)
            // Windows prevents truncating a file while its mapping is open.
            // Force genuine misses even after the dense hybrid test populated
            // the same cache with other expert IDs.
            disk.stage(group, {0, 1, 2, 3});
            std::filesystem::resize_file(gguf, 0);
            const std::vector<int32_t> unread{8, 9, 8, 9};
            ggml_backend_tensor_set(ids, unread.data(), 0, unread.size() * sizeof(int32_t));
            rejected = false;
            try { ggml_backend_sched_graph_compute(scheduler.get(), graph); }
            catch (const std::runtime_error & e) {
                if (pin_experts) throw;
                rejected = std::string(e.what()).find("failed to read selected expert") != std::string::npos;
                if (!rejected) throw;
            }
            if (pin_experts) {
                const auto actual = core::read_tensor_f32(cached);
                const auto reference = core::read_tensor_f32(expected);
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i]) || std::abs(actual[i] - reference[i]) > 1e-4F)
                        throw std::runtime_error("pinned experts did not retain their loaded snapshot");
                }
            } else if (!rejected) throw std::runtime_error("failed expert reads were not propagated");
#endif
            core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        std::filesystem::resize_file(gguf, 0);
        bool rejected = false;
        try { core::DiskExpertWeightStore truncated(gguf, false, 1, cpu_execution); }
        catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("truncated expert GGUF header was not rejected");
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
