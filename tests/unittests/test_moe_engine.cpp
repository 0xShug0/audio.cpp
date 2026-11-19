#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/moe_engine/moe_prepared_graph.h"
#include "test_allocation_failure.h"
#include "../../src/framework/moe_engine/moe_engine_internal.h"
#include "../../external/ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

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
const std::vector<std::string> names{"model.layers.0.mlp.experts.up.weight", "model.layers.0.mlp.experts.gate.weight", "model.layers.0.mlp.experts.down.weight"};
moe::LayerBindings bindings() {
    moe::LayerBindings result;
    result.experts = {
        {"model.layers.0.mlp.experts.up.weight", "model.layers.0.mlp.experts.up.weight", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Up},
        {"model.layers.0.mlp.experts.gate.weight", "model.layers.0.mlp.experts.gate.weight", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Gate},
        {"model.layers.0.mlp.experts.down.weight", "model.layers.0.mlp.experts.down.weight", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Down}};
    return result;
}
moe::LayerMath mathematics() {
    moe::LayerMath result; result.routing.experts = experts; result.routing.selected = selected; return result;
}

// Independent readback dependency contract. Async downloads are withheld until
// synchronization; synchronous downloads reject an unfinished producer. This
// checks ordering without assuming a tiny GPU copy remains physically running.
struct QueuedReadback {
    ggml_backend_t backend;
    ggml_tensor * watched;
    ggml_backend_buffer_t buffer;
    decltype(ggml_backend_i::get_tensor_async) original_get;
    decltype(ggml_backend_i::synchronize) original_sync;
    decltype(ggml_backend_i::graph_compute) original_compute;
    decltype(ggml_backend_buffer_i::get_tensor) original_buffer_get;
    void * destination = nullptr;
    size_t offset = 0, bytes = 0, completed = 0;
    bool producer_pending = false, delivering = false;
    static QueuedReadback * active;
    QueuedReadback(ggml_backend_t backend, ggml_tensor * watched)
        : backend(backend), watched(watched), buffer(watched->buffer),
          original_get(backend->iface.get_tensor_async), original_sync(backend->iface.synchronize),
          original_compute(backend->iface.graph_compute), original_buffer_get(buffer->iface.get_tensor) {
        require(!active, "nested queued readback fixture");
        active = this;
        backend->iface.graph_compute = [](ggml_backend_t value, ggml_cgraph * graph) {
            active->producer_pending = true;
            return active->original_compute(value, graph);
        };
        buffer->iface.get_tensor = [](ggml_backend_buffer_t value, const ggml_tensor * tensor,
            void * output, size_t offset, size_t bytes) {
            auto & self = *active;
            if (tensor == self.watched && !self.delivering) {
                require(!self.producer_pending, "native routing read IDs before producer completion");
                ++self.completed;
            }
            self.original_buffer_get(value, tensor, output, offset, bytes);
        };
        backend->iface.get_tensor_async = [](ggml_backend_t value, const ggml_tensor * tensor,
            void * output, size_t offset, size_t bytes) {
            auto & self = *active;
            if (tensor == self.watched) {
                require(!self.destination, "overlapping watched route downloads");
                self.destination = output; self.offset = offset; self.bytes = bytes;
            } else if (self.original_get) self.original_get(value, tensor, output, offset, bytes);
            else {
                if (self.original_sync) self.original_sync(value);
                self.producer_pending = false;
                ggml_backend_tensor_get(tensor, output, offset, bytes);
            }
        };
        backend->iface.synchronize = [](ggml_backend_t value) {
            auto & self = *active;
            self.delivering = true;
            if (self.destination && !self.original_get) {
                // Preserve GGML's shared-memory fallback: drain before reading.
                if (self.original_sync) self.original_sync(value);
                self.producer_pending = false;
                ggml_backend_tensor_get(self.watched, self.destination, self.offset, self.bytes);
            } else {
                if (self.destination) self.original_get(value, self.watched, self.destination, self.offset, self.bytes);
                if (self.original_sync) self.original_sync(value);
                self.producer_pending = false;
            }
            self.delivering = false;
            if (self.destination) { self.destination = nullptr; ++self.completed; }
        };
    }
    ~QueuedReadback() {
        buffer->iface.get_tensor = original_buffer_get;
        backend->iface.graph_compute = original_compute;
        backend->iface.get_tensor_async = original_get;
        backend->iface.synchronize = original_sync;
        active = nullptr;
    }
};
QueuedReadback * QueuedReadback::active = nullptr;

// Public engine integration versus a separately built native GGML reference.
// Route changes exceed four slots and revisit evicted experts. Independent graph
// work remains live around embedded execution to detect scratch corruption.
void compare(core::ExecutionContext & execution, const assets::TensorSource & source,
    moe::EngineConfig config, int64_t tokens, bool phased,
    moe::ExpertLowering lowering = moe::ExpertLowering::Default) {
    core::BackendWeightStore weights(execution.backend(), execution.backend_type(), "test.moe.engine.weights", 1024 * 1024);
    moe::MoeEngine engine(execution, weights, source, config);
    const auto layer = engine.register_layer(bindings(), mathematics());
    // Staged banks and auxiliary-CPU banks belong to the engine. This fixture
    // has no unrelated backbone weights to upload into the caller's store.
    const bool primary_resident = config.residency == moe::Residency::Resident &&
        (core::is_host_backend(execution.backend()) ||
         std::find(config.cpu_layers.begin(), config.cpu_layers.end(), size_t{0}) == config.cpu_layers.end());
    if (primary_resident) weights.upload();
    engine.finalize_weights();
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
        const auto call = engine.add_call(layer, {tokens, width, width, 32, GGML_TYPE_I32, 1, lowering}, workspace);
        engine.build_inputs(call, *owner, 0, x, &routing);
        actual = engine.build_result(call, *owner);
        sequence = owner->add_sequence({size_t{0}, call, finish});
    } else actual = engine.build(*owner, layer, x, &routing, lowering);
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
    std::unique_ptr<QueuedReadback> queued_ids;
    if (config.residency == moe::Residency::Staged && tokens == 1 && !phased &&
        config.policies.placement.name == "default") {
        ggml_tensor * mapped = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(owner->graph()); ++i) {
            auto * node = ggml_graph_node(owner->graph(), i);
            if (node->op == GGML_OP_MUL_MAT_ID) { mapped = node->src[2]; break; }
        }
        require(mapped && mapped != ids, "native routing must materialize cache slot IDs");
        queued_ids = std::make_unique<QueuedReadback>(execution.backend(), mapped);
    }
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
        const auto reads_before = queued_ids ? queued_ids->completed : size_t{0};
        if (phased) owner->compute(sequence); else owner->compute();
        // A scheduler may also copy the watched IDs between backends. Require
        // a completed read in this request, not one particular read API/count.
        if (queued_ids) require(!queued_ids->destination && queued_ids->completed > reads_before,
            "native routing did not complete its queued ID download before execution");
        reference.compute();
        const auto a = core::read_tensor_f32(result), b = core::read_tensor_f32(expected);
        require(a.size() == b.size(), "engine/reference output shape differs");
        for (size_t i = 0; i < a.size(); ++i)
            require(std::isfinite(a[i]) && std::abs(a[i] - b[i]) <= 1e-6F * (1 + std::abs(b[i])), "engine differs from independent native expert graph");
        require(engine.statistics().buffer_bytes == baseline_bytes, "expert weight buffers grew during reuse");
        const auto live = core::read_tensor_f32(unrelated);
        for (size_t i = 0; i < live.size(); ++i) require(live[i] == input[i] * 0.03125F, "embedded expert scratch overwrote parent-live data");
    }
    queued_ids.reset(); reference.release(); owner.reset(); engine.release_execution();
    require(engine.statistics().buffer_bytes == baseline_bytes, "execution release discarded expert weight registration");
}

// Cache-slot assignment is part of the legacy native indexed-matmul path.
// Use the standard container's iteration contract as the independent oracle;
// hardcoding its order would make the fixture standard-library dependent.
void cache_assignment_order(core::ExecutionContext & execution, const assets::TensorSource & source) {
    detail::ExpertStorage storage(source.source_path(), false, 1, execution.backend(), moe::prepare_policy({}).eviction);
    const auto group = storage.add_group(names, execution.backend(), 4);
    for (const std::vector<int32_t> & ids : {std::vector<int32_t>{2, 0, 2, 1}, {9, 1, 4, 7}, {11, 3}}) {
        storage.invalidate_cache();
        const std::unordered_set<int32_t> unique(ids.begin(), ids.end());
        const std::vector<int32_t> slot_contents(unique.begin(), unique.end());
        std::vector<int32_t> expected;
        for (auto id : ids)
            expected.push_back(static_cast<int32_t>(std::find(slot_contents.begin(), slot_contents.end(), id) - slot_contents.begin()));
        require(storage.stage(group, ids) == expected, "default staging changed legacy cache-slot assignment order");
        require(storage.stage(group, ids) == expected, "cache hits changed legacy cache-slot assignment order");
    }
}

// Each token routes to the same two experts, so these are actual per-expert
// assignment counts, not total token counts spread across a larger bank.
void bucket_boundaries(core::ExecutionContext & execution, const assets::TensorSource & source,
    const std::vector<core::TensorValue> & native) {
    const auto policy = moe::prepare_policy({});
    detail::ExpertStorage storage(source.source_path(), false, 1, execution.backend(), policy.eviction);
    const auto group = storage.add_group(names, execution.backend(), 4);
    for (const int64_t tokens : {31, 32, 33, 65}) {
        auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
        detail::ProgramShape shape;
        shape.hidden_size = width; shape.input_size = width; shape.tokens = tokens;
        shape.selected = selected; shape.experts = experts; shape.cache_slots = 4;
        shape.rows_per_expert = {32}; shape.scatter_index_type = GGML_TYPE_I32;
        detail::ExpertProgram program(execution, shape, workspace, nullptr, policy);
        const detail::ExpertWeights bank{storage.tensor(group, 0).tensor,
            storage.tensor(group, 1).tensor, storage.tensor(group, 2).tensor};
        const auto layer = program.add_layer({}, bank, &storage, group, execution.backend(), nullptr);
        core::PreparedGraph producer(execution, {"test.moe.bucket.inputs", 1024 * 1024, 256, 32 * ggml_tensor_overhead()});
        auto * persistent = producer.persistent_build_context().ggml;
        auto * x = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, width, tokens);
        auto * ids = ggml_new_tensor_2d(persistent, GGML_TYPE_I32, selected, tokens);
        auto * mix = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, selected, tokens);
        for (auto * tensor : {x, ids, mix}) ggml_set_input(tensor);
        producer.allocate_persistent();
        program.build_inputs(producer.build_context(), producer.graph(), x, {ids, mix});
        core::PreparedGraph finish(execution, {"test.moe.bucket.finish", 1024 * 1024, 256});
        auto * combined = detail::combine(finish.build_context(), program.output(), program.routing_weights());
        ggml_set_output(combined); ggml_build_forward_expand(finish.graph(), combined);
        workspace->add(producer); workspace->add(finish); workspace->prepare(); program.prepare();
        core::PreparedGraph reference(execution, {"test.moe.bucket.reference", 1024 * 1024, 256});
        auto * g = reference.build_context().ggml;
        auto * rows = ggml_reshape_3d(g, x, width, 1, tokens);
        auto * up = ggml_mul_mat_id(g, native[0].tensor, rows, ids);
        auto * gate = ggml_silu(g, ggml_mul_mat_id(g, native[1].tensor, rows, ids));
        auto * expected = ggml_mul_mat_id(g, native[2].tensor, ggml_mul(g, gate, up), ids);
        auto * weighted = ggml_mul(g, expected, ggml_reshape_3d(g, mix, 1, selected, tokens));
        auto * expected_sum = ggml_reshape_2d(g,
            ggml_sum_rows(g, ggml_cont(g, ggml_permute(g, weighted, 1, 0, 2, 3))), width, tokens);
        ggml_set_output(expected); ggml_set_output(expected_sum);
        ggml_build_forward_expand(reference.graph(), expected_sum); reference.allocate();
        std::vector<float> input(static_cast<size_t>(width * tokens));
        std::vector<int32_t> routes(static_cast<size_t>(selected * tokens));
        std::vector<float> mixing(routes.size());
        for (size_t i = 0; i < mixing.size(); ++i) mixing[i] = i % selected ? 0.375F : 0.625F;
        // Change both inputs and logical experts, exceed cache capacity, then
        // revisit the initial bank to exercise reloads at each bucket boundary.
        for (int repeat = 0; repeat < 4; ++repeat) {
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = 0.5F * std::sin(static_cast<float>(i) * 0.17F + repeat);
            const int32_t first = repeat == 3 ? 0 : repeat * 2;
            for (size_t i = 0; i < routes.size(); ++i) routes[i] = first + static_cast<int32_t>(i % selected);
            for (const int32_t expert : {first, first + 1})
                require(std::count(routes.begin(), routes.end(), expert) == tokens, "bucket fixture lost concentrated assignment counts");
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
            ggml_backend_tensor_set(mix, mixing.data(), 0, mixing.size() * sizeof(float));
            const auto before = storage.statistics();
            if (repeat == 3) {
                const auto & resident = storage.cached_experts(group);
                require(std::find(resident.begin(), resident.end(), 0) == resident.end() &&
                    std::find(resident.begin(), resident.end(), 1) == resident.end(),
                    "bucket revisit did not actually evict its original experts");
            }
            producer.compute();
            require(program.run(layer) == static_cast<uint64_t>(((tokens + 31) / 32) * 32 * shape.cache_slots),
                "bucket boundary executed an unexpected number of padded expert rows");
            if (repeat == 3)
                require(storage.statistics().expert_loads == before.expert_loads + 2,
                    "bucket revisit did not reload both evicted expert banks");
            finish.compute();
            reference.compute();
            const auto actual = core::read_tensor_f32(program.output());
            const auto wanted = core::read_tensor_f32(expected);
            require(actual.size() == wanted.size(), "bucket boundary output shape differs");
            float signal = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                signal = std::max(signal, std::abs(wanted[i]));
                require(std::isfinite(actual[i]) && std::abs(actual[i] - wanted[i]) <= 1e-6F * (1 + std::abs(wanted[i])),
                    "bucket boundary differs from independent native expert graph");
            }
            require(signal > 1e-4F, "bucket reference is too small to detect missing expert output");
            const auto reduced = core::read_tensor_f32(combined);
            const auto wanted_sum = core::read_tensor_f32(expected_sum);
            require(reduced.size() == wanted_sum.size(), "bucket reduction output shape differs");
            for (size_t i = 0; i < reduced.size(); ++i)
                require(std::isfinite(reduced[i]) && std::abs(reduced[i] - wanted_sum[i]) <= 1e-6F * (1 + std::abs(wanted_sum[i])),
                    "bucket boundary changed weighted assignment reduction");
        }
        std::cout << "bucket assignments_per_expert=" << tokens << " native_type=" << ggml_type_name(native[0].tensor->type) << '\n';
    }
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


// Fractional placement is inapplicable to a CPU bank, whether the primary
// backend is CPU or this is a whole-layer CPU offload in a GPU session.
void cpu_placement_precedence(core::ExecutionContext & execution, const assets::TensorSource & source,
    const std::vector<core::TensorValue> & reference_weights, bool native_single_token) {
    std::unique_ptr<core::ExecutionContext> auxiliary;
    auto * host = &execution;
    if (!core::is_host_backend(execution.backend())) {
        core::BackendConfig config; config.threads = 8;
        auxiliary = std::make_unique<core::ExecutionContext>(config); host = auxiliary.get();
    }
    moe::PolicyConfig selection;
    selection.placement = {"placement_cpu_share", {{"share", "1"}}};
    const auto policy = moe::prepare_policy(selection);
    auto placement = policy.placement(static_cast<size_t>(experts));
    detail::ExpertStorage storage(source.source_path(), false, 1, execution.backend(), policy.eviction);
    const auto group = storage.add_group(names, host->backend(), 4);
    auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
    detail::ProgramShape shape;
    shape.identity = 23; shape.hidden_size = width; shape.input_size = width;
    shape.tokens = 1; shape.selected = selected; shape.experts = experts; shape.cache_slots = 4;
    shape.native_single_token = native_single_token;
    shape.rows_per_expert = {1};
    detail::ExpertProgram program(execution, shape, workspace, host, policy);
    const detail::ExpertWeights bank{storage.tensor(group, 0).tensor, storage.tensor(group, 1).tensor, storage.tensor(group, 2).tensor};
    const auto layer = program.add_layer({}, bank, &storage, group, host->backend(), placement.get());
    core::PreparedGraph producer(execution, {"test.moe.cpu_precedence", 1024 * 1024, 256, 32 * ggml_tensor_overhead()});
    auto * persistent = producer.persistent_build_context().ggml;
    auto * x = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, width, 1);
    auto * ids = ggml_new_tensor_2d(persistent, GGML_TYPE_I32, selected, 1);
    auto * mix = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, selected, 1);
    for (auto * tensor : {x, ids, mix}) ggml_set_input(tensor);
    producer.allocate_persistent();
    program.build_inputs(producer.build_context(), producer.graph(), x, {ids, mix});
    workspace->add(producer); workspace->prepare(); program.prepare();
    core::PreparedGraph reference(execution, {"test.moe.cpu_precedence.reference", 1024 * 1024, 256});
    auto * g = reference.build_context().ggml;
    auto * rows = ggml_reshape_3d(g, x, width, 1, 1);
    auto * up = ggml_mul_mat_id(g, reference_weights[0].tensor, rows, ids);
    auto * gate = ggml_silu(g, ggml_mul_mat_id(g, reference_weights[1].tensor, rows, ids));
    auto * expected = ggml_mul_mat_id(g, reference_weights[2].tensor, ggml_mul(g, gate, up), ids);
    ggml_set_output(expected); ggml_build_forward_expand(reference.graph(), expected); reference.allocate();
    std::vector<float> input(static_cast<size_t>(width), 0.01F);
    const std::vector<float> mixing(static_cast<size_t>(selected), 0.5F);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(mix, mixing.data(), 0, mixing.size() * sizeof(float));
    // Fill four cache slots, then force misses at share=1. Whole-layer CPU
    // placement must honor the requested native or bucketed execution layout.
    for (const std::vector<int32_t> & route : {std::vector<int32_t>{0, 1}, {2, 3}, {4, 5}, {0, 1}}) {
        ggml_backend_tensor_set(ids, route.data(), 0, route.size() * sizeof(int32_t));
        producer.compute();
        const auto expected_rows = native_single_token ? selected : shape.cache_slots;
        require(program.run(layer) == static_cast<uint64_t>(expected_rows), "CPU-selected single-token experts changed the requested execution layout");
        reference.compute();
        const auto actual = core::read_tensor_f32(program.output()), oracle = core::read_tensor_f32(expected);
        for (size_t element = 0; element < actual.size(); ++element)
            require(std::isfinite(actual[element]) && std::abs(actual[element] - oracle[element]) <= 1e-6F * (1 + std::abs(oracle[element])),
                "CPU placement precedence changed expert output");
    }
    reference.release(); workspace->reset(); producer.release();
}

struct ObservationState {
    std::vector<moe::ExecutionObservation> completed;
    std::vector<moe::PolicyNotification> placement_events, cache_events;
    size_t accesses = 0, selections = 0;
    uint64_t epoch = 0;
    bool context_valid = true, invalidation_empty = true;
};
class CountPlacement final : public moe::PlacementPolicy {
public:
    explicit CountPlacement(ObservationState & state) : state_(state) {}
    void select(const moe::PlacementInput & input, moe::PlacementDecision & decision) override {
        decision.cpu.clear(); decision.gpu.clear();
        // Preserve fixed hits before misses, without consulting the production policy.
        for (bool hits : {true, false}) for (auto expert : input.experts)
            if ((std::find(input.resident.begin(), input.resident.end(), expert) != input.resident.end()) == hits)
                decision.gpu.push_back(expert);
    }
    void observe(const moe::ExecutionObservation & observation) override { state_.completed.push_back(observation); }
    void event(const moe::PolicyNotification & event) override { state_.placement_events.push_back(event); }
private:
    ObservationState & state_;
};
class ContextEviction final : public moe::EvictionPolicy {
public:
    explicit ContextEviction(ObservationState & state) : state_(state) {}
    size_t select(const moe::EvictionInput & input) const override {
        state_.context_valid = state_.context_valid && input.cache.group == 0 && input.cache.epoch == state_.epoch &&
            input.cache.resident.size() == 4 && !input.candidates.empty();
        ++state_.selections;
        return std::min_element(input.candidates.begin(), input.candidates.end(), [](const auto & a, const auto & b) {
            return a.last_used < b.last_used;
        })->slot;
    }
    void accessed(const moe::CacheAccess & access, const moe::CacheContext & cache) override {
        state_.context_valid = state_.context_valid && cache.group == 0 && cache.epoch == state_.epoch &&
            access.expert >= 0 && access.expert < experts;
        state_.accesses += access.assignments;
    }
    void event(const moe::PolicyNotification & notification, const moe::CacheContext & cache) override {
        state_.cache_events.push_back(notification);
        state_.context_valid = state_.context_valid && notification.cache_epoch == cache.epoch && cache.group == 0;
        if (notification.event == moe::PolicyEvent::CacheInvalidated) {
            state_.epoch = cache.epoch;
            state_.invalidation_empty = state_.invalidation_empty &&
                std::all_of(cache.resident.begin(), cache.resident.end(), [](int32_t expert) { return expert == -1; });
        }
    }
private:
    ObservationState & state_;
};

// A counts-only policy needs neither CPU placement nor phase clocks. The
// executor must deliver completed rows even though phase_times is false.
void policy_observations(core::ExecutionContext & execution, const assets::TensorSource & source) {
    ObservationState state;
    state.completed.reserve(4); state.placement_events.reserve(8); state.cache_events.reserve(8);
    moe::PreparedPolicy policy;
    policy.requirements.completed_work = true;
    policy.requirements.cache_accesses = true;
    policy.eviction = [&](size_t) { return std::make_unique<ContextEviction>(state); };
    detail::ExpertStorage storage(source.source_path(), false, 1, execution.backend(), policy.eviction);
    const auto group = storage.add_group(names, execution.backend(), 4);
    storage.event({moe::PolicyEvent::GenerationBegin, 0, storage.cache_epoch()});
    {
        auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
        CountPlacement placement(state);
        detail::ProgramShape shape;
        shape.identity = 17; shape.hidden_size = width; shape.input_size = width;
        shape.tokens = 7; shape.selected = selected; shape.experts = experts; shape.cache_slots = 4;
        shape.rows_per_expert = {1, 2, 4, 8};
        detail::ExpertProgram program(execution, shape, workspace, nullptr, policy);
        const detail::ExpertWeights bank{storage.tensor(group, 0).tensor, storage.tensor(group, 1).tensor, storage.tensor(group, 2).tensor};
        const auto layer = program.add_layer({}, bank, &storage, group, execution.backend(), &placement);
        core::PreparedGraph producer(execution, {"test.moe.observations", 1024 * 1024, 256, 32 * ggml_tensor_overhead()});
        auto * persistent = producer.persistent_build_context().ggml;
        auto * x = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, width, shape.tokens);
        auto * ids = ggml_new_tensor_2d(persistent, GGML_TYPE_I32, selected, shape.tokens);
        auto * mix = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, selected, shape.tokens);
        for (auto * t : {x, ids, mix}) ggml_set_input(t);
        producer.allocate_persistent();
        program.build_inputs(producer.build_context(), producer.graph(), x, {ids, mix});
        workspace->add(producer); workspace->prepare(); program.prepare();
        std::vector<float> input(static_cast<size_t>(width * shape.tokens), 0.01F);
        std::vector<float> weights(static_cast<size_t>(selected * shape.tokens), 0.5F);
        std::vector<int32_t> routes(weights.size());
        for (size_t i = 0; i < routes.size(); ++i) routes[i] = static_cast<int32_t>(i % static_cast<size_t>(experts));
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mix, weights.data(), 0, weights.size() * sizeof(float));
        for (int repeat = 0; repeat < 2; ++repeat) {
            producer.compute(); program.run(layer);
            if (repeat == 0) storage.invalidate_cache();
        }
        require(state.completed.size() == 2, "counts-only policy did not receive successful observations");
        for (size_t request = 0; request < state.completed.size(); ++request) {
            const auto & observation = state.completed[request];
            require(observation.cache_epoch == request, "completed observation lost its cache epoch");
            require(observation.cpu_rows + observation.gpu_rows == routes.size() && observation.program == shape.identity,
                "completed-work counts/program identity differ from the controlled assignments");
            require((core::is_host_backend(execution.backend()) ? observation.cpu_rows : observation.gpu_rows) == routes.size(),
                "observation attributed completed rows to the wrong backend");
            require(!observation.has_phase_times && observation.cpu_ms == 0 && observation.gpu_ms == 0 && observation.upload_ms == 0,
                "counts-only observations implicitly enabled phase timing");
            require(observation.uploaded_bytes > 0, "counts-only observations omitted actual transfer accounting");
        }
        const auto completed_accesses = state.accesses;
        auto invalid = routes; invalid.front() = static_cast<int32_t>(experts);
        bool rejected = false;
        try { program.run(layer, invalid); } catch (const std::out_of_range &) { rejected = true; }
        require(rejected && state.completed.size() == 2 && state.accesses == completed_accesses,
            "failed work became a completed observation/cache access");
        require(state.placement_events.size() == 1 && state.placement_events.back().event == moe::PolicyEvent::ExecutionFailed &&
            state.placement_events.back().program == shape.identity && state.placement_events.back().cache_epoch == storage.cache_epoch(),
            "placement failure lost program/cache identity");
        workspace->reset(); producer.release();
    }
    storage.event({moe::PolicyEvent::ExecutionReleased, 0, storage.cache_epoch()});
    const std::vector<moe::PolicyEvent> expected{moe::PolicyEvent::GenerationBegin, moe::PolicyEvent::CacheInvalidated,
        moe::PolicyEvent::ExecutionFailed, moe::PolicyEvent::ProgramRetired, moe::PolicyEvent::ExecutionReleased};
    require(state.cache_events.size() == expected.size(), "cache policy missed lifecycle notifications");
    require(state.cache_events[2].program == 17 && state.cache_events[2].cpu_rows == 0 && state.cache_events[2].gpu_rows == 0,
        "cache failure notification lost identity or counted unexecuted assignments");
    for (size_t i = 0; i < expected.size(); ++i) require(state.cache_events[i].event == expected[i], "cache lifecycle event ordering changed");
    require(state.context_valid && state.invalidation_empty && state.epoch == 1 && state.selections > 0 && state.accesses == 28,
        "cache policy did not receive current residency, epoch or completed accesses");
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
            assets::GgufConversionOptions conversion;
            conversion.type_overrides.push_back({"*", type});
            assets::convert_tensor_sources_to_gguf({{tensor_path, ""}}, path, type, true, false,
                {}, {}, std::nullopt, std::move(conversion));
            auto source = assets::open_tensor_source(path);
            for (const auto & name : names)
                require(assets::resolve_tensor_storage_type(*source, name, assets::TensorStorageType::Native) == type,
                    "expert fixture conversion did not preserve its requested native precision");
            core::BackendWeightStore reference_store(execution.backend(), execution.backend_type(), "test.moe.storage.reference", 1024 * 1024);
            std::vector<core::TensorValue> reference;
            for (const auto & name : names)
                reference.push_back(reference_store.load_tensor(*source, name, assets::TensorStorageType::Native, {experts, width, width}));
            require(reference.front().tensor->type == assets::ggml_type_for_tensor_storage(type),
                "expert reference bank is not using the requested native precision");
            reference_store.upload();
            for (bool pinned : {false, true}) {
                if (pinned && !ggml_backend_dev_host_buffer_type(ggml_backend_get_device(execution.backend()))) continue;
                check_group_registration_failures(path, execution, names, pinned);
                check_host_bank_failures(path, execution, names, reference, pinned);
            }
            cache_assignment_order(execution, *source);
            if (type == assets::TensorStorageType::F32) policy_observations(execution, *source);
            for (bool staged : {false, true}) for (bool cpu_layer : {false, true}) {
                moe::EngineConfig mode;
                mode.residency = staged ? moe::Residency::Staged : moe::Residency::Resident;
                mode.cache_slots = 4;
                if (cpu_layer) mode.cpu_layers = {0};
                compare(execution, *source, mode, 1, false);
                compare(execution, *source, mode, 7, true);
                if (staged) {
                    compare(execution, *source, mode, 7, false);
                    compare(execution, *source, mode, 1, true);
                }
            }
            bucket_boundaries(execution, *source, reference);
            // Existing CUDA fused expert lowering is a separate graph contract,
            // not a new placement policy. Keep the ordinary native-bank oracle.
            if (execution.backend_type() == core::BackendType::Cuda) {
                moe::EngineConfig fused;
                compare(execution, *source, fused, 1, false, moe::ExpertLowering::FusedSwiGlu);
                fused.residency = moe::Residency::Staged; fused.cache_slots = 4;
                compare(execution, *source, fused, 1, false, moe::ExpertLowering::FusedSwiGlu);
                compare(execution, *source, fused, 7, true, moe::ExpertLowering::FusedSwiGlu);
                std::cout << "fused expert lowering native_type=" << ggml_type_name(reference.front().tensor->type) << '\n';
            }
            for (bool native_single_token : {false, true})
                cpu_placement_precedence(execution, *source, reference, native_single_token);
            moe::EngineConfig cpu_share; cpu_share.residency = moe::Residency::Staged; cpu_share.cache_slots = 4;
            cpu_share.policies.placement = {"placement_cpu_share", {{"share", "1"}}};
            if (!core::is_host_backend(execution.backend())) cpu_share.cpu_layers = {0};
            compare(execution, *source, cpu_share, 1, false);
            compare(execution, *source, cpu_share, 7, true);
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
