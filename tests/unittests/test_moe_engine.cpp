#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/moe_engine/moe_prepared_graph.h"
#include "test_allocation_failure.h"
#include "../../src/framework/moe_engine/moe_engine_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <tuple>

namespace {
namespace core = engine::core;
namespace assets = engine::assets;
namespace moe = engine::moe;
namespace detail = engine::moe::detail;

void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
struct FixtureDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("audiocpp_moe_engine_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FixtureDirectory() { std::filesystem::create_directories(path); }
    ~FixtureDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
constexpr int64_t width = 256, experts = 12, selected = 2;
const std::vector<std::string> names{"up", "gate", "down"};
moe::LayerBindings bindings() {
    moe::LayerBindings result;
    result.experts = {
        {"up", "up", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Up},
        {"gate", "gate", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Gate},
        {"down", "down", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Down}};
    return result;
}
moe::LayerMath mathematics() {
    moe::LayerMath result; result.routing.experts = experts; result.routing.selected = selected; return result;
}

// Public engine integration versus a separately built native GGML reference.
// Route changes exceed four slots and revisit evicted experts. Independent graph
// work remains live around embedded execution to detect scratch corruption.
void compare(core::ExecutionContext & execution, const assets::TensorSource & source,
    moe::EngineConfig config, int64_t tokens, bool phased) {
    core::BackendWeightStore weights(execution.backend(), execution.backend_type(), "test.moe.engine.weights", 1024 * 1024);
    moe::MoeEngine engine(execution, weights, source, config);
    const auto layer = engine.register_layer(bindings(), mathematics());
    weights.upload(); engine.finalize_weights();
    core::BackendWeightStore reference_weights(execution.backend(), execution.backend_type(), "test.moe.reference.weights", 1024 * 1024);
    std::vector<core::TensorValue> native;
    for (const auto & name : names) native.push_back(reference_weights.load_tensor(source, name, assets::TensorStorageType::Native, {experts, width, width}));
    reference_weights.upload();
    const auto baseline_bytes = engine.statistics().buffer_bytes;
    auto workspace = phased ? std::make_shared<core::GraphWorkspace>(execution.backend()) : nullptr;
    auto owner = std::make_unique<moe::MoePreparedGraph>(execution,
        core::PreparedGraphConfig{"test.moe.engine", 4 * 1024 * 1024, 1024, 64 * ggml_tensor_overhead()}, workspace);
    auto & inputs = owner->persistent_build_context();
    auto * x = ggml_new_tensor_2d(inputs.ggml, GGML_TYPE_F32, width, tokens);
    auto * ids = ggml_new_tensor_2d(inputs.ggml, GGML_TYPE_I32, selected, tokens);
    auto * mix = ggml_new_tensor_2d(inputs.ggml, GGML_TYPE_F32, selected, tokens);
    for (auto * input : {x, ids, mix}) ggml_set_input(input);
    owner->allocate_persistent();
    moe::Routing routing{ids, mix};
    auto & ctx = owner->build_context();
    const auto finish = phased ? owner->add_graph("finish") : size_t{0};
    ggml_tensor * actual = nullptr;
    moe::MoePreparedGraph::SequenceId sequence{0};
    if (phased) {
        const auto call = engine.add_call(layer, {tokens, width, width, 32, GGML_TYPE_I32}, workspace);
        engine.build_inputs(call, *owner, 0, x, &routing);
        actual = engine.build_result(call, *owner);
        sequence = owner->add_sequence({size_t{0}, call, finish});
    } else actual = engine.build(*owner, layer, x, &routing);
    auto * unrelated = ggml_scale(ctx.ggml, x, 0.03125F);
    auto * result = ggml_add(ctx.ggml, actual, unrelated);
    ggml_set_output(result); ggml_set_output(unrelated);
    ggml_build_forward_expand(owner->graph(finish), unrelated);
    ggml_build_forward_expand(owner->graph(finish), result);
    owner->allocate();

    core::PreparedGraph reference(execution, {"test.moe.independent", 2 * 1024 * 1024, 256});
    auto * g = reference.build_context().ggml;
    auto * rows = ggml_reshape_3d(g, x, width, 1, tokens);
    auto * gate = ggml_mul_mat_id(g, native[1].tensor, rows, ids);
    auto * up = ggml_mul_mat_id(g, native[0].tensor, rows, ids);
    auto * lanes = ggml_mul_mat_id(g, native[2].tensor, ggml_mul(g, ggml_silu(g, gate), up), ids);
    auto * scaled = ggml_mul(g, lanes, ggml_reshape_3d(g, mix, 1, selected, tokens));
    auto * sum = ggml_reshape_2d(g, ggml_sum_rows(g, ggml_cont(g, ggml_permute(g, scaled, 1, 0, 2, 3))), width, tokens);
    auto * expected = ggml_add(g, sum, ggml_scale(g, x, 0.03125F));
    ggml_set_output(expected); ggml_build_forward_expand(reference.graph(), expected); reference.allocate();
    std::vector<float> input(static_cast<size_t>(width * tokens)), mixing(static_cast<size_t>(selected * tokens));
    std::vector<int32_t> routes(static_cast<size_t>(selected * tokens));
    for (int repeat = 0; repeat < 6; ++repeat) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = 0.01F * std::sin(static_cast<float>(i) * 0.17F + static_cast<float>(repeat));
        for (size_t i = 0; i < routes.size(); ++i) {
            routes[i] = static_cast<int32_t>((i + static_cast<size_t>(repeat) * 3) % static_cast<size_t>(experts));
            mixing[i] = (i % 2) ? 0.375F : 0.625F;
        }
        if (repeat == 3) engine.begin_generation();
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mix, mixing.data(), 0, mixing.size() * sizeof(float));
        if (phased) owner->compute(sequence); else owner->compute();
        reference.compute();
        const auto a = core::read_tensor_f32(result), b = core::read_tensor_f32(expected);
        require(a.size() == b.size(), "engine/reference output shape differs");
        for (size_t i = 0; i < a.size(); ++i)
            require(std::isfinite(a[i]) && std::abs(a[i] - b[i]) <= 1e-6F * (1 + std::abs(b[i])), "engine differs from independent native expert graph");
        require(engine.statistics().buffer_bytes == baseline_bytes, "expert weight buffers grew during reuse");
        const auto live = core::read_tensor_f32(unrelated);
        for (size_t i = 0; i < live.size(); ++i) require(live[i] == input[i] * 0.03125F, "embedded expert scratch overwrote parent-live data");
    }
    reference.release(); owner.reset(); engine.release_execution();
    require(engine.statistics().buffer_bytes == baseline_bytes, "execution release discarded expert weight registration");
}

void check_group_registration_failures(const std::filesystem::path & path,
    engine::core::ExecutionContext & execution, const std::vector<std::string> & names,
    bool pinned) {
    namespace core = engine::core;
    size_t failures = 0;
    bool completed = false;
    for (size_t allocation = 0; allocation < 64; ++allocation) {
        {
            detail::ExpertStorage store(path, pinned, 1, execution.backend(), moe::prepare_policy({}).eviction);
            const auto first = store.add_group(names, execution.backend(), 2);
            const auto slots = store.stage(first, {0, 1});
            const auto before = store.statistics();
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
    const std::vector<std::string> & names, const std::vector<engine::core::TensorValue> & reference, bool pinned) {
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
            detail::ExpertStorage store(path, pinned, 1, execution.backend(), moe::prepare_policy({}).eviction);
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


}

int main(int argc, char ** argv) {
    const auto start = std::chrono::steady_clock::now();
    try {
        engine::debug::configure_logging({true, std::nullopt});
        core::BackendConfig config; config.threads = 8;
        if (argc > 1 && std::string(argv[1]) == "cuda") config.type = core::BackendType::Cuda;
        if (argc > 1 && std::string(argv[1]) == "vulkan") config.type = core::BackendType::Vulkan;
        if (argc > 1 && std::string(argv[1]) == "metal") config.type = core::BackendType::Metal;
        core::ExecutionContext execution(config);
        FixtureDirectory directory;
        std::vector<engine::io::SafeTensorWriteEntry> entries;
        for (size_t projection = 0; projection < names.size(); ++projection) {
            std::vector<float> values(static_cast<size_t>(experts * width * width));
            for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(static_cast<float>(i) * 0.013F + static_cast<float>(projection)) * 0.01F;
            std::vector<unsigned char> bytes(values.size() * sizeof(float)); std::memcpy(bytes.data(), values.data(), bytes.size());
            entries.push_back({names[projection], "F32", {experts, width, width}, std::move(bytes)});
        }
        const auto tensor_path = directory.path / "weights.safetensors";
        engine::io::write_safetensors_file(tensor_path, entries);
        for (auto type : {assets::TensorStorageType::F32, assets::TensorStorageType::Q4_K}) {
            const auto path = directory.path / "weights.gguf";
            assets::convert_tensor_sources_to_gguf({{tensor_path, ""}}, path, type, true, false);
            auto source = assets::open_tensor_source(path);
            core::BackendWeightStore reference_store(execution.backend(), execution.backend_type(), "test.moe.storage.reference", 1024 * 1024);
            std::vector<core::TensorValue> reference;
            for (const auto & name : names)
                reference.push_back(reference_store.load_tensor(*source, name, assets::TensorStorageType::Native, {experts, width, width}));
            reference_store.upload();
            for (bool pinned : {false, true}) {
                if (pinned && !ggml_backend_dev_host_buffer_type(ggml_backend_get_device(execution.backend()))) continue;
                check_group_registration_failures(path, execution, names, pinned);
                check_host_bank_failures(path, execution, names, reference, pinned);
            }
            for (bool staged : {false, true}) for (bool cpu_layer : {false, true}) {
                moe::EngineConfig mode;
                mode.residency = staged ? moe::Residency::Staged : moe::Residency::Resident;
                mode.cache_slots = 4;
                if (cpu_layer) mode.cpu_layers = {0};
                compare(execution, *source, mode, 1, false);
                compare(execution, *source, mode, 7, true);
                if (staged) compare(execution, *source, mode, 7, false);
            }
            if (!core::is_host_backend(execution.backend())) {
                moe::EngineConfig hybrid; hybrid.residency = moe::Residency::Staged; hybrid.cache_slots = 4;
                hybrid.policies.placement = {"placement_cpu_share", {{"share", "0.5"}}};
                compare(execution, *source, hybrid, 7, false); compare(execution, *source, hybrid, 7, true);
            }
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        std::cout << "moe_engine: PASS\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
