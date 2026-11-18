#include "engine/framework/moe_engine/moe_prepared_graph.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"
#include "ggml-backend-impl.h"
#include "test_assert.h"

#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
namespace core = engine::core;
namespace moe = engine::moe;
namespace assets = engine::assets;
using engine::test::require;
constexpr int64_t width = 256, experts = 4, selected = 2;

struct Weights {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("moe_graph_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::shared_ptr<const assets::TensorSource> source;
    std::array<std::vector<float>, 3> native;
    explicit Weights(assets::TensorStorageType type = assets::TensorStorageType::F32) {
        std::filesystem::create_directories(directory);
        std::vector<engine::io::SafeTensorWriteEntry> entries;
        const std::array<const char *, 3> names{"up", "gate", "down"};
        for (size_t projection = 0; projection < names.size(); ++projection) {
            std::vector<float> values(static_cast<size_t>(experts * width * width), 0);
            for (int64_t expert = 0; expert < experts; ++expert)
                for (int64_t channel = 0; channel < width; ++channel)
                    values[static_cast<size_t>((expert * width + channel) * width + channel)] =
                        static_cast<float>((projection + 1) * static_cast<size_t>(expert + 1)) / 16;
            std::vector<unsigned char> bytes(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            entries.push_back({names[projection], "F32", {experts, width, width}, std::move(bytes)});
        }
        const auto tensors = directory / "weights.safetensors";
        engine::io::write_safetensors_file(tensors, entries);
        const auto gguf = directory / "weights.gguf";
        assets::convert_tensor_sources_to_gguf({{tensors, ""}}, gguf, type, true, false);
        source = assets::open_tensor_source(gguf);
        for (size_t projection = 0; projection < names.size(); ++projection)
            native[projection] = source->require_f32(names[projection], {experts, width, width});
    }
    ~Weights() { source.reset(); std::error_code error; std::filesystem::remove_all(directory, error); }
    moe::LayerBindings bindings() const {
        moe::LayerBindings result;
        result.experts = {
            {"up", "up", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Up},
            {"gate", "gate", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Gate},
            {"down", "down", {experts, width, width}, assets::TensorStorageType::Native, moe::ProjectionRole::Down}};
        return result;
    }
};

// A deterministic rendezvous at a real scheduler evaluation boundary. The
// timeout only prevents a hung test; overlap is never inferred from a sleep.
struct Rendezvous {
    std::mutex mutex;
    std::condition_variable changed;
    size_t arrived = 0;
    bool released = false, cancelled = false;
    void enter() {
        std::unique_lock<std::mutex> lock(mutex);
        ++arrived; changed.notify_all();
        if (!changed.wait_for(lock, std::chrono::seconds(30), [&] { return released || cancelled; }))
            throw std::runtime_error("scheduler rendezvous timed out");
        if (cancelled) throw std::runtime_error("scheduler rendezvous cancelled");
    }
    void wait(size_t count) {
        std::unique_lock<std::mutex> lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(30), [&] { return arrived >= count || cancelled; }),
            "scheduler did not reach its evaluation boundary");
        require(!cancelled, "a concurrent graph failed before the rendezvous");
    }
    void release(bool cancel = false) {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = cancel; released = true; changed.notify_all();
    }
};

struct Evaluation {
    ggml_tensor * watched = nullptr;
    Rendezvous * gate = nullptr;
    size_t completed = 0;
    bool fail = false;
    static bool callback(ggml_tensor * tensor, bool ask, void * data) {
        auto & self = *static_cast<Evaluation *>(data);
        if (tensor != self.watched) return false;
        if (ask) return true;
        ++self.completed;
        if (self.gate) self.gate->enter();
        if (self.fail) throw std::runtime_error("injected parent evaluation failure");
        return true;
    }
};

struct Program {
    std::unique_ptr<moe::MoePreparedGraph> graph;
    ggml_tensor * x = nullptr, * ids = nullptr, * mix = nullptr, * output = nullptr;
    Evaluation evaluation;
    moe::MoePreparedGraph::SequenceId sequence{0};
    int64_t tokens;
    bool phased;
    const Weights & reference;
    Program(core::ExecutionContext & execution, core::ExecutionContext & fallback, moe::MoeEngine & engine,
        moe::LayerId layer, int64_t count, bool phases, const Weights & fixture)
        : tokens(count), phased(phases), reference(fixture) {
        auto workspace = phases ? std::make_shared<core::GraphWorkspace>(execution.backend()) : nullptr;
        graph = std::make_unique<moe::MoePreparedGraph>(execution,
            core::PreparedGraphConfig{"test.moe.graph", 4 * 1024 * 1024, 1024, 64 * ggml_tensor_overhead()}, workspace);
        auto * persistent = graph->persistent_build_context().ggml;
        x = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, width, tokens);
        ids = ggml_new_tensor_2d(persistent, GGML_TYPE_I32, selected, tokens);
        mix = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, selected, tokens);
        for (auto * input : {x, ids, mix}) ggml_set_input(input);
        graph->allocate_persistent();
        auto & ctx = graph->build_context();
        moe::Routing routing{ids, mix};
        ggml_tensor * routed;
        auto * producer = ggml_scale(ctx.ggml, x, 1);
        ggml_set_output(producer);
        if (phases) {
            const auto call = engine.add_call(layer, {tokens, width, width}, workspace);
            engine.build_inputs(call, *graph, 0, producer, &routing);
            const auto finish = graph->add_graph("consumer");
            routed = engine.build_result(call, *graph);
            output = ggml_add(ctx.ggml, routed, ggml_scale(ctx.ggml, x, 0.125F));
            ggml_set_output(output); ggml_build_forward_expand(graph->graph(finish), output);
            sequence = graph->add_sequence({size_t{0}, call, finish});
        } else {
            ggml_build_forward_expand(graph->graph(), producer);
            routed = engine.build(*graph, layer, producer, &routing);
            output = ggml_add(ctx.ggml, routed, ggml_scale(ctx.ggml, x, 0.125F));
            ggml_set_output(output); ggml_build_forward_expand(graph->graph(), output);
            evaluation.watched = producer;
            moe::GraphRequirements requirements;
            requirements.scheduler = true;
            requirements.backends = {execution.backend(), fallback.backend()};
            requirements.placements = {{producer, execution.backend()}, {output, execution.backend()}};
            requirements.evaluation = Evaluation::callback;
            requirements.evaluation_data = &evaluation;
            graph->declare_execution(std::move(requirements));
        }
        graph->allocate();
    }
    std::vector<float> input(int seed) const {
        std::vector<float> values(static_cast<size_t>(width * tokens));
        for (size_t i = 0; i < values.size(); ++i) values[i] = 0.1F * std::sin(static_cast<float>(i) * 0.13F + seed);
        return values;
    }
    void upload(int seed) {
        const auto values = input(seed);
        std::vector<int32_t> routes(static_cast<size_t>(selected * tokens));
        std::vector<float> mixing(routes.size());
        for (size_t i = 0; i < routes.size(); ++i) {
            routes[i] = static_cast<int32_t>((i + static_cast<size_t>(seed)) % static_cast<size_t>(experts));
            mixing[i] = i % 2 ? 0.375F : 0.625F;
        }
        ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mix, mixing.data(), 0, mixing.size() * sizeof(float));
    }
    void compute() { if (phased) graph->compute(sequence); else graph->compute(); }
    std::vector<float> check(int seed) {
        const auto values = core::read_tensor_f32(output);
        const auto in = input(seed);
        std::vector<float> expected(in.size());
        std::array<float, static_cast<size_t>(width)> middle{};
        // Scalar arithmetic over the actual native/dequantized banks. Do not
        // assume quantization preserves the source's diagonal coefficients.
        for (size_t token = 0; token < static_cast<size_t>(tokens); ++token) {
            for (size_t channel = 0; channel < static_cast<size_t>(width); ++channel)
                expected[token * width + channel] = in[token * width + channel] * 0.125F;
            for (size_t lane = 0; lane < static_cast<size_t>(selected); ++lane) {
                const auto expert = (token * static_cast<size_t>(selected) + lane + static_cast<size_t>(seed)) % static_cast<size_t>(experts);
                for (size_t row = 0; row < static_cast<size_t>(width); ++row) {
                    float up = 0, gate = 0;
                    for (size_t column = 0; column < static_cast<size_t>(width); ++column) {
                        const auto offset = (expert * width + row) * width + column;
                        up += in[token * width + column] * reference.native[0][offset];
                        gate += in[token * width + column] * reference.native[1][offset];
                    }
                    middle[row] = (gate / (1 + std::exp(-gate))) * up;
                }
                for (size_t row = 0; row < static_cast<size_t>(width); ++row) {
                    float output = 0;
                    for (size_t column = 0; column < static_cast<size_t>(width); ++column)
                        output += middle[column] * reference.native[2][(expert * width + row) * width + column];
                    expected[token * width + row] += (lane ? 0.375F : 0.625F) * output;
                }
            }
        }
        for (size_t i = 0; i < values.size(); ++i)
            require(std::isfinite(values[i]) && std::abs(values[i] - expected[i]) <= 1e-6F * (1 + std::abs(expected[i])),
                "MoE graph differs from independent native-bank scalar expert/ordered-reduction oracle");
        return values;
    }
};

struct Session {
    core::ExecutionContext execution, fallback;
    core::BackendWeightStore weights;
    moe::MoeEngine engine;
    moe::LayerId layer;
    const Weights & reference;
    Session(const core::BackendConfig & backend, const Weights & fixture, bool hybrid = false, bool cpu_layer = false, bool resident = false)
        : execution(backend), fallback(core::BackendConfig{}),
          weights(execution.backend(), execution.backend_type(), "test.moe.graph.weights", 1024 * 1024),
          engine(execution, weights, *fixture.source, configuration(hybrid, cpu_layer, resident)), reference(fixture) {
        moe::LayerMath math; math.routing.experts = experts; math.routing.selected = selected;
        layer = engine.register_layer(fixture.bindings(), math);
        weights.upload(); engine.finalize_weights();
    }
    static moe::EngineConfig configuration(bool hybrid, bool cpu_layer, bool resident) {
        moe::EngineConfig config; config.residency = resident ? moe::Residency::Resident : moe::Residency::Staged; config.cache_slots = 2;
        if (hybrid) config.policies.placement = {"placement_cpu_share", {{"share", "1"}}};
        if (cpu_layer) config.cpu_layers = {0};
        return config;
    }
    std::unique_ptr<Program> program(int64_t tokens, bool phased = false) {
        return std::make_unique<Program>(execution, fallback, engine, layer, tokens, phased, reference);
    }
};

void callback_rejection(core::ExecutionContext & execution) {
    auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
    moe::MoePreparedGraph graph(execution, {"test.moe.shared.callback", 65536, 32}, workspace);
    moe::GraphRequirements requirements; requirements.evaluation = Evaluation::callback;
    Evaluation evaluation; requirements.evaluation_data = &evaluation;
    graph.declare_execution(requirements);
    for (bool shared : {false, true}) {
        bool rejected = false;
        try { if (shared) moe::MoePreparedGraph::allocate_shared({&graph}); else graph.allocate(); }
        catch (const std::invalid_argument & error) {
            rejected = std::string(error.what()) == "graph evaluation callback requires an explicit scheduler";
        }
        require(rejected, "shared/direct allocation silently accepted an unsupported evaluation callback");
    }
    graph.reset();
    auto * g = graph.build_context().ggml;
    auto * input = ggml_new_tensor_1d(g, GGML_TYPE_F32, 32);
    ggml_set_input(input);
    auto * output = ggml_scale(g, input, 2); ggml_set_output(output); ggml_build_forward_expand(graph.graph(), output);
    moe::MoePreparedGraph::allocate_shared({&graph});
    std::array<float, 32> values{}; for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i);
    ggml_backend_tensor_set(input, values.data(), 0, sizeof(values)); graph.compute();
    const auto actual = core::read_tensor_f32(output);
    for (size_t i = 0; i < values.size(); ++i) require(actual[i] == 2 * values[i], "rejected shared preparation poisoned retry");
    graph.release();
}

void requirement_conflicts(core::ExecutionContext & execution, core::ExecutionContext & fallback) {
    moe::MoePreparedGraph graph(execution, {"test.moe.requirements", 65536, 64});
    auto * g = graph.build_context().ggml;
    auto * x = ggml_new_tensor_1d(g, GGML_TYPE_F32, 32);
    auto * output = ggml_scale(g, x, 2);
    graph.place_nodes({output}, execution.backend());
    bool rejected = false;
    try { graph.place_nodes({output}, fallback.backend()); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "graph accepted conflicting explicit placements for the same node");

    auto workspace = std::make_shared<core::GraphWorkspace>(execution.backend());
    moe::MoePreparedGraph shared(execution, {"test.moe.requirements.shared", 65536, 64}, workspace);
    moe::GraphRequirements requirements; requirements.scheduler = true;
    requirements.backends = {execution.backend(), fallback.backend()};
    shared.declare_execution(requirements);
    rejected = false;
    try { shared.allocate(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "scheduler-backed graph silently joined a serial shared workspace");
    rejected = false;
    try { moe::MoePreparedGraph::allocate_shared({&shared}); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "shared allocation silently discarded scheduler requirements");

    moe::MoePreparedGraph entries(execution, {"test.moe.requirements.entries", 65536, 64});
    entries.add_entry_point("body", 0);
    entries.declare_execution(requirements);
    rejected = false;
    try { entries.allocate(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "scheduled parent accepted unsupported multiple entry points");
}

void resident_cpu_callback(const core::BackendConfig & backend, const Weights & fixture) {
    Session session(backend, fixture, false, true, true);
    auto program = session.program(3);
    program->upload(1); program->compute(); program->check(1);
    program->upload(2); program->compute(); program->check(2);
    require(program->evaluation.completed == 2, "CPU-resident expert requirements discarded the backbone callback");
    require(session.engine.statistics().buffer_bytes == 0, "CPU-resident layer acquired a bounded staged cache");
}

void overlap_and_callback(Session & session) {
    auto first = session.program(3), second = session.program(3);
    first->upload(1); second->upload(2);
    Rendezvous gate; first->evaluation.gate = &gate;
    auto running = std::async(std::launch::async, [&] {
        try { first->compute(); } catch (...) { gate.release(true); throw; }
    });
    try {
        gate.wait(1);
        size_t rejected = 0;
        const auto attempt = [&](auto && action) {
            try { action(); } catch (const std::runtime_error &) { ++rejected; } catch (const std::logic_error &) { ++rejected; }
        };
        attempt([&] { first->compute(); });
        attempt([&] { second->compute(); });
        attempt([&] { first->graph->reset(); });
        attempt([&] { first->graph->release(); });
        attempt([&] { session.engine.begin_generation(); });
        attempt([&] { session.engine.release_execution(); });
        require(rejected == 6, "active engine/graph sequence permitted overlap or lifecycle mutation");
        require(second->evaluation.completed == 0, "second graph submitted producers before engine lease rejection");
        gate.release(); running.get();
    } catch (...) { gate.release(true); running.wait(); throw; }
    first->evaluation.gate = nullptr;
    first->check(1); second->compute(); second->check(2);
    require(first->evaluation.completed == 1 && second->evaluation.completed == 1,
        "scheduler lost or duplicated ordinary evaluation callbacks around expert replacement");
    first->upload(3); first->compute(); first->check(3);
    require(first->evaluation.completed == 2, "scheduler callback was lost on graph reuse");
}

void lifecycle(Session & session) {
    const auto bytes = session.engine.statistics().buffer_bytes;
    auto x = session.program(3, true);
    x->upload(1); x->compute(); const auto original = x->check(1);
    auto y = session.program(5, true);
    y->upload(2); y->compute(); y->check(2);
    // Keep X bound while Y is prepared/executed: returning to an existing
    // program must refresh routing/cache contents without rebuilding X.
    x->upload(1); x->compute();
    require(x->check(1) == original, "return to a retained shape X changed output");
    y.reset(); x.reset();
    session.engine.release_execution(); session.engine.begin_generation();
    auto rebuilt = session.program(3, true); rebuilt->upload(1); rebuilt->compute();
    require(rebuilt->check(1) == original, "release/reprepare changed shape X output");
    require(session.engine.statistics().buffer_bytes == bytes, "expert caches grew after shape replacement");
}

void independent_sessions(const core::BackendConfig & backend, const Weights & fixture) {
    Session left(backend, fixture), right(backend, fixture);
    auto a = left.program(3), b = right.program(3); a->upload(1); b->upload(2);
    Rendezvous gate; a->evaluation.gate = b->evaluation.gate = &gate;
    const auto work = [&](Program & program) {
        try { program.compute(); } catch (...) { gate.release(true); throw; }
    };
    auto first = std::async(std::launch::async, [&] { work(*a); });
    auto second = std::async(std::launch::async, [&] { work(*b); });
    try { gate.wait(2); gate.release(); first.get(); second.get(); }
    catch (...) { gate.release(true); first.wait(); second.wait(); throw; }
    a->evaluation.gate = b->evaluation.gate = nullptr;
    require(a->check(1) != b->check(2), "independent sessions returned the same stale input/output");
}

// Intercept only the primary backend boundaries. A GPU merge failure is injected
// after its CPU-result upload was actually queued, not before execution starts.
struct FailureCheck {
    ggml_backend_t backend;
    decltype(ggml_backend_i::graph_compute) original_compute;
    decltype(ggml_backend_i::set_tensor_async) original_upload;
    decltype(ggml_backend_i::synchronize) original_sync;
    ggml_tensor * uploaded = nullptr;
    bool armed = false, failed = false, alive_on_drain = true;
    size_t drains = 0;
    static FailureCheck * active;
    explicit FailureCheck(ggml_backend_t value) : backend(value), original_compute(value->iface.graph_compute),
        original_upload(value->iface.set_tensor_async), original_sync(value->iface.synchronize) {
        require(!active, "nested backend failure observer");
        require(original_compute && original_upload, "post-upload failure fixture requires an asynchronous GPU upload boundary");
        active = this;
        backend->iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * t, const void * data, size_t offset, size_t size) {
            auto & self = *active;
            self.original_upload(b, t, data, offset, size);
            if (self.armed && t->type == GGML_TYPE_F32 && t->ne[0] == width && t->ne[1] == 6 && t->ne[2] == 1)
                self.uploaded = t;
        };
        backend->iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * graph) {
            auto & self = *active;
            bool scatter = false, matmul = false;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const auto op = ggml_graph_node(graph, i)->op;
                scatter = scatter || op == GGML_OP_SET_ROWS;
                matmul = matmul || op == GGML_OP_MUL_MAT_ID;
            }
            if (self.armed && self.uploaded && scatter && !matmul) {
                self.failed = true; self.armed = false; return GGML_STATUS_FAILED;
            }
            return self.original_compute(b, graph);
        };
        backend->iface.synchronize = [](ggml_backend_t b) {
            auto & self = *active;
            if (self.failed) {
                ++self.drains;
                self.alive_on_drain = self.alive_on_drain && self.uploaded && self.uploaded->buffer && self.uploaded->data;
            }
            if (self.original_sync) self.original_sync(b);
        };
    }
    ~FailureCheck() {
        backend->iface.graph_compute = original_compute;
        backend->iface.set_tensor_async = original_upload;
        backend->iface.synchronize = original_sync; active = nullptr;
    }
};
FailureCheck * FailureCheck::active = nullptr;

void failure_recovery(Session & session) {
    auto program = session.program(3);
    program->upload(1); program->compute(); const auto original = program->check(1);
    // CPU primary execution has no hybrid upload; still exercise exceptional
    // parent-graph exit and subsequent engine lease/resource reuse.
    if (core::is_host_backend(session.execution.backend())) {
        program->evaluation.fail = true;
        bool rejected = false;
        try { program->compute(); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "injected callback failure was ignored");
        program->evaluation.fail = false;
    } else {
        FailureCheck failure(session.execution.backend()); failure.armed = true;
        program->upload(1);
        bool rejected = false;
        try { program->compute(); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected && failure.uploaded && failure.failed, "did not inject failure after the real CPU-result upload");
        require(failure.drains > 0 && failure.alive_on_drain, "failed work was not drained before releasing live upload storage");
    }
    program->upload(1); program->compute();
    require(program->check(1) == original, "failed execution poisoned the next request or retained its lease");
    program.reset(); session.engine.release_execution();
    auto rebuilt = session.program(3); rebuilt->upload(1); rebuilt->compute();
    require(rebuilt->check(1) == original, "failed execution poisoned release/reprepare");
}
} // namespace

int main(int argc, char ** argv) {
    const auto start = std::chrono::steady_clock::now();
    try {
        engine::debug::configure_logging({true, std::nullopt});
        core::BackendConfig backend; backend.threads = 8;
        if (argc > 1 && std::string(argv[1]) == "cuda") backend.type = core::BackendType::Cuda;
        if (argc > 1 && std::string(argv[1]) == "vulkan") backend.type = core::BackendType::Vulkan;
        Weights fixture;
        Session session(backend, fixture);
        callback_rejection(session.execution); requirement_conflicts(session.execution, session.fallback);
        resident_cpu_callback(backend, fixture); overlap_and_callback(session); lifecycle(session);
        independent_sessions(backend, fixture);
        Session hybrid(backend, fixture, !core::is_host_backend(session.execution.backend()));
        failure_recovery(hybrid);
        Weights quantized(assets::TensorStorageType::Q4_K);
        Session native_quantized(backend, quantized);
        lifecycle(native_quantized);
        Session hybrid_quantized(backend, quantized, !core::is_host_backend(session.execution.backend()));
        failure_recovery(hybrid_quantized);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        std::cout << "moe_prepared_graph: PASS\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
