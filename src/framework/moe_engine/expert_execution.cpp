#include "moe_engine_internal.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <variant>

namespace engine::moe::detail {
class ExpertProgram::Impl {
public:
    // Bucket views must satisfy the buffer provider's offset alignment. Slot
    // IDs are constants, encoded and uploaded once during program preparation.
    struct SlotLayout {
        std::vector<size_t> offsets;
        std::vector<int32_t> ids;
        SlotLayout() = default;
        SlotLayout(ggml_backend_t backend, const ProgramShape & config) {
            const auto alignment = ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
            offsets.reserve(config.rows_per_expert.size());
            for (auto count : config.rows_per_expert) {
                const auto bytes = ids.size() * sizeof(int32_t);
                const auto aligned = ((bytes + alignment - 1) / alignment) * alignment;
                offsets.push_back(aligned);
                const auto first = aligned / sizeof(int32_t);
                const auto rows = static_cast<size_t>(count);
                ids.resize(first + static_cast<size_t>(config.cache_slots) * rows);
                for (int64_t slot = 0; slot < config.cache_slots; ++slot)
                    std::fill_n(ids.begin() + first + static_cast<size_t>(slot) * rows, rows, static_cast<int32_t>(slot));
            }
        }
    };
    struct Layer {
        ExpertStorage * store;
        size_t group;
        std::vector<ggml_cgraph *> graphs;
        std::vector<std::unique_ptr<core::HostGraphPlan>> plans;
        std::vector<core::PreparedGraph::GraphId> cpu_graphs;
        std::vector<ggml_tensor *> cpu_outputs;
        PlacementPolicy * placement = nullptr;
        PlacementDecision decision;
        bool cpu_staged = false, direct = false, place_experts = false;
    };
    struct Hybrid {
        core::ExecutionContext & execution;
        core::PreparedGraph graphs;
        ggml_tensor * hidden = nullptr;
        ggml_tensor * source = nullptr;
        ggml_tensor * ids = nullptr;
        std::vector<int64_t> buckets;
        std::vector<ggml_cgraph *> scatter;
        std::vector<std::unique_ptr<core::HostGraphPlan>> scatter_plans;
        std::vector<int32_t> source_rows, expert_ids;
        std::variant<std::vector<int32_t>, std::vector<int64_t>> target_rows;
        std::mutex mutex;
        std::condition_variable ready, finished;
        std::thread worker;
        bool stopping = false, pending = false;
        Layer * layer = nullptr;
        size_t bucket = 0;
        std::exception_ptr error;
        double compute_ms = 0;
        bool measure_times;

        Hybrid(core::ExecutionContext & cpu_execution, const ProgramShape & config, bool timing)
            : execution(cpu_execution),
              graphs(execution, {"framework.moe.cpu", 16 * 1024 * 1024, 128, 64 * ggml_tensor_overhead()}), measure_times(timing) {
            const auto rows = config.tokens * config.selected;
            for (int64_t count = 1; count < rows; count *= 2) buckets.push_back(count);
            buckets.push_back(rows);
            auto & ctx = graphs.persistent_build_context();
            hidden = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, config.input_size, config.tokens);
            source = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, rows);
            ids = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_I32, 1, rows);
            for (auto * tensor : {hidden, source, ids}) ggml_set_input(tensor);
            graphs.allocate_persistent();
            source_rows.resize(static_cast<size_t>(rows));
            expert_ids.resize(static_cast<size_t>(rows));
            if (config.scatter_index_type == GGML_TYPE_I64) target_rows.emplace<std::vector<int64_t>>();
            std::visit([rows](auto & targets) { targets.resize(static_cast<size_t>(rows)); }, target_rows);
        }
        ~Hybrid() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            ready.notify_one();
            if (worker.joinable()) worker.join();
        }
        void start() {
            worker = std::thread([this] {
                std::unique_lock<std::mutex> lock(mutex);
                for (;;) {
                    ready.wait(lock, [this] { return pending || stopping; });
                    if (stopping && !pending) return;
                    lock.unlock();
                    const auto started = measure_times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    try { graphs.compute(layer->cpu_graphs[bucket]); }
                    catch (...) { error = std::current_exception(); }
                    compute_ms = measure_times ? debug::elapsed_ms(started) : 0;
                    lock.lock();
                    pending = false;
                    finished.notify_one();
                }
            });
        }
        void launch(Layer & next_layer, size_t next_bucket) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                layer = &next_layer;
                bucket = next_bucket;
                error = nullptr;
                pending = true;
            }
            ready.notify_one();
        }
        void wait() {
            std::unique_lock<std::mutex> lock(mutex);
            finished.wait(lock, [this] { return !pending; });
            if (error) std::rethrow_exception(error);
        }
    };
    struct CpuStage {
        core::PreparedGraph graphs;
        SlotLayout slot_layout;
        ggml_tensor * hidden = nullptr;
        ggml_tensor * source = nullptr;
        ggml_tensor * slots = nullptr;
        bool has_graphs = false;
        std::vector<float> batch_output, ordered_output;

        CpuStage(core::ExecutionContext & execution, const ProgramShape & config)
            : graphs(execution, {"framework.moe.cpu_staged", 16 * 1024 * 1024, 128, 64 * ggml_tensor_overhead()}),
              slot_layout(execution.backend(), config) {
            const auto rows = config.resident ? config.selected * config.tokens : config.cache_slots * config.rows_per_expert.back();
            auto & ctx = graphs.persistent_build_context();
            hidden = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, config.input_size, config.tokens);
            source = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, rows);
            slots = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_I32, 1, static_cast<int64_t>(slot_layout.ids.size()));
            for (auto * tensor : {hidden, source, slots}) ggml_set_input(tensor);
            graphs.allocate_persistent();
            ggml_backend_tensor_set(slots, slot_layout.ids.data(), 0, slot_layout.ids.size() * sizeof(int32_t));
            batch_output.resize(static_cast<size_t>(config.hidden_size * rows));
            ordered_output.resize(static_cast<size_t>(config.hidden_size * config.selected * config.tokens));
        }
    };
    const core::ExecutionContext & execution;
    ProgramShape config;
    bool hybrid_enabled = false;
    PolicyRequirements requirements;
    std::vector<size_t> assignment_counts;
    std::vector<int32_t> experts, group, policy_experts;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free}, inputs{nullptr, ggml_free};
    std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::shared_ptr<core::GraphWorkspace> workspace;
    std::vector<Layer> layers;
    bool prepared = false;
    ggml_tensor * hidden = nullptr, * ids = nullptr, * weights = nullptr, * outputs = nullptr, * result = nullptr;
    ggml_tensor * source = nullptr, * target = nullptr, * slots = nullptr;
    SlotLayout slot_layout;
    std::vector<int32_t> logical_ids, source_rows;
    std::variant<std::vector<int32_t>, std::vector<int64_t>> target_rows;
    std::vector<size_t> assignment_offsets, assignment_cursor;
    std::vector<int32_t> assignment_positions;
    std::vector<uint8_t> completed, placement_destinations;
    core::ExecutionContext * cpu_execution;
    std::unique_ptr<CpuStage> cpu_stage;
    std::unique_ptr<Hybrid> hybrid;
    ggml_tensor * cpu_output = nullptr, * cpu_target = nullptr;
    Statistics stats;

    uint64_t run_direct(Layer & binding);
    uint64_t run_cpu_staged(Layer & binding, const std::vector<int32_t> & experts);
    uint64_t run_gpu(Layer & binding, const std::vector<int32_t> & experts);
    uint64_t run(Layer & binding);
    uint64_t run_hybrid(Layer & binding, const PlacementDecision & placement);

    Impl(const core::ExecutionContext & e, ProgramShape c, std::shared_ptr<core::GraphWorkspace> shared_workspace,
        core::ExecutionContext * cpu_execution, const PreparedPolicy & policy)
        : execution(e), config(std::move(c)), requirements(policy.requirements), workspace(std::move(shared_workspace)), cpu_execution(cpu_execution) {
        if (!workspace) throw std::invalid_argument("staged MoE requires a caller-owned graph workspace");
        hybrid_enabled = !core::is_host_backend(execution.backend()) && requirements.cpu_execution;
        if (hybrid_enabled && (!cpu_execution || !core::is_host_backend(cpu_execution->backend())))
            throw std::invalid_argument("hybrid MoE requires a borrowed host execution context");
        if (config.input_size == 0) config.input_size = config.hidden_size;
        if (config.hidden_size <= 0 || config.input_size <= 0 || config.tokens <= 0 || config.selected <= 0 || config.experts < config.selected ||
            config.cache_slots <= 0 || config.cache_slots > config.experts || config.rows_per_expert.empty() ||
            (config.scatter_index_type != GGML_TYPE_I32 && config.scatter_index_type != GGML_TYPE_I64))
            throw std::invalid_argument("invalid staged MoE configuration");
        int64_t previous = 0;
        for (auto count : config.rows_per_expert) {
            if (count <= previous) throw std::invalid_argument("staged MoE row buckets must increase");
            previous = count;
        }
        if (config.cache_slots > std::numeric_limits<int32_t>::max() / previous)
            throw std::invalid_argument("staged MoE row buckets exceed indexed graph limits");
        const auto rows = config.resident ? config.selected * config.tokens : config.cache_slots * previous;
        if (config.tokens > std::numeric_limits<int32_t>::max() / config.selected ||
            rows > std::numeric_limits<int32_t>::max() - config.tokens * config.selected)
            throw std::invalid_argument("staged MoE assignments exceed indexed graph limits");
        if (hybrid_enabled && config.tokens * config.selected > std::numeric_limits<int32_t>::max() / 2)
            throw std::invalid_argument("hybrid MoE assignments exceed indexed graph limits");
        slot_layout = SlotLayout(execution.backend(), config);
        inputs.reset(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
        context.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (!inputs || !context) throw std::runtime_error("cannot allocate staged MoE contexts");
        auto * g = inputs.get();
        hidden = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.input_size, config.tokens);
        ids = ggml_new_tensor_2d(g, GGML_TYPE_I32, config.selected, config.tokens);
        weights = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.selected, config.tokens);
        const auto padding_rows = hybrid_enabled ? std::max(rows, config.selected * config.tokens) : rows;
        outputs = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.hidden_size, config.selected * config.tokens + padding_rows);
        source = ggml_new_tensor_1d(g, GGML_TYPE_I32, rows);
        target = ggml_new_tensor_1d(g, config.scatter_index_type, rows);
        slots = ggml_new_tensor_2d(g, GGML_TYPE_I32, 1, static_cast<int64_t>(slot_layout.ids.size()));
        result = ggml_view_3d(g, outputs, config.hidden_size, config.selected, config.tokens,
            config.hidden_size * sizeof(float), config.hidden_size * config.selected * sizeof(float), 0);
        for (auto * t : {hidden, ids, weights, outputs, source, target, slots}) ggml_set_input(t);
        if (hybrid_enabled) {
            cpu_output = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.hidden_size, config.selected * config.tokens);
            cpu_target = ggml_new_tensor_1d(g, config.scatter_index_type, config.selected * config.tokens);
            ggml_set_input(cpu_output);
            ggml_set_input(cpu_target);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (!buffer) throw std::runtime_error("cannot allocate staged MoE buffers");
        ggml_backend_buffer_clear(buffer.get(), 0);
        logical_ids.resize(static_cast<size_t>(config.selected * config.tokens));
        source_rows.resize(static_cast<size_t>(rows));
        if (config.scatter_index_type == GGML_TYPE_I64) target_rows.emplace<std::vector<int64_t>>();
        std::visit([rows](auto & targets) { targets.resize(static_cast<size_t>(rows)); }, target_rows);
        ggml_backend_tensor_set(slots, slot_layout.ids.data(), 0, slot_layout.ids.size() * sizeof(int32_t));
        assignment_offsets.resize(static_cast<size_t>(config.experts) + 1);
        assignment_cursor.resize(static_cast<size_t>(config.experts));
        assignment_positions.resize(static_cast<size_t>(config.selected * config.tokens));
        assignment_counts.resize(static_cast<size_t>(config.experts));
        if (requirements.cache_accesses) completed.resize(static_cast<size_t>(config.selected * config.tokens));
        if (hybrid_enabled) placement_destinations.resize(static_cast<size_t>(config.experts));
        experts.reserve(static_cast<size_t>(config.experts));
        policy_experts.reserve(static_cast<size_t>(config.experts));
        group.reserve(static_cast<size_t>(config.cache_slots));
    }

    ~Impl() {
        // Join CPU work before freeing graphs or borrowed expert views.
        const bool has_cpu_execution = hybrid || cpu_stage;
        if (hybrid) {
            auto scatter = std::move(hybrid->scatter);
            hybrid.reset();
            for (auto * graph : scatter) core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        cpu_stage.reset();
        if (has_cpu_execution) {
            debug::timing_log_scalar("framework.moe.cpu_assignments", stats.cpu_assignments);
            debug::timing_log_scalar("framework.moe.gpu_assignments", stats.gpu_assignments);
            if (requirements.phase_times) debug::timing_log_scalar("framework.moe.cpu_ms", stats.cpu_ms);
        }
        for (auto & layer : layers) {
            const PolicyNotification retired{PolicyEvent::ProgramRetired, config.identity,
                layer.store ? layer.store->cache_epoch() : uint64_t{0}};
            if (layer.placement) layer.placement->event(retired);
            if (layer.store) layer.store->event(layer.group, retired);
        }
        for (auto & layer : layers)
            for (auto * graph : layer.graphs) core::release_backend_graph_resources(execution.backend(), graph, true);
    }
};

ExpertProgram::ExpertProgram(const core::ExecutionContext & execution, ProgramShape config,
    std::shared_ptr<core::GraphWorkspace> workspace, core::ExecutionContext * cpu_execution, const PreparedPolicy & policy)
    : impl_(std::make_unique<Impl>(execution, std::move(config), std::move(workspace), cpu_execution, policy)) {}
ExpertProgram::~ExpertProgram() = default;

size_t ExpertProgram::add_layer(const ExpertMath & math, const ExpertWeights & weights,
    ExpertStorage * store, size_t group, ggml_backend_t backend, PlacementPolicy * placement) {
    auto & s = *impl_;
    if (s.prepared) throw std::logic_error("cannot add staged MoE layers after preparation");
    if (weights.up->ne[2] != s.config.cache_slots || weights.down->ne[1] != s.config.hidden_size)
        throw std::invalid_argument("staged MoE expert bank does not match cache configuration");
    Impl::Layer layer;
    layer.store = store;
    layer.group = group;
    layer.cpu_staged = backend != s.execution.backend() && core::is_host_backend(backend);
    // CPU-only and whole-layer CPU execution do not split work across devices.
    // Keep the policy binding for requested observations, not placement.
    layer.place_experts = placement && !core::is_host_backend(backend);
    // Preserve the caller's execution layout. Embedded native one-token graphs
    // use top-k matmuls; phased graphs keep their cache-slot row buckets.
    layer.direct = s.config.resident || (s.config.native_single_token && s.config.tokens == 1 && s.config.rows_per_expert.front() == 1 &&
        s.config.selected <= s.config.cache_slots && !layer.place_experts);
    if (layer.cpu_staged) {
        if (!s.cpu_execution || s.cpu_execution->backend() != backend)
            throw std::invalid_argument("staged CPU MoE requires the cache's borrowed host execution context");
        if (!s.cpu_stage) s.cpu_stage = std::make_unique<Impl::CpuStage>(*s.cpu_execution, s.config);
        auto & host = *s.cpu_stage;
        auto & ctx = host.graphs.build_context();
        if (layer.direct) {
            const auto id = !host.has_graphs ? size_t{0}
                : host.graphs.add_graph("layer_" + std::to_string(s.layers.size()) + "_direct");
            host.has_graphs = true;
            auto * ids = ggml_view_2d(ctx.ggml, host.source, s.config.selected, s.config.tokens,
                s.config.selected * sizeof(int32_t), 0);
            auto * input = ggml_reshape_3d(ctx.ggml, host.hidden, s.config.input_size, 1, s.config.tokens);
            auto * output = build_experts(ctx, input, ids, weights, math);
            ggml_set_output(output);
            ggml_build_forward_expand(host.graphs.graph(id), output);
            core::validate_backend_graph_supported(backend, host.graphs.graph(id), "staged CPU MoE experts");
            layer.cpu_graphs.push_back(id);
            layer.cpu_outputs.push_back(output);
            s.layers.push_back(std::move(layer));
            return s.layers.size() - 1;
        }
        size_t bucket = 0;
        for (auto count : s.config.rows_per_expert) {
            const auto rows = count * s.config.cache_slots;
            const auto id = !host.has_graphs ? size_t{0}
                : host.graphs.add_graph("layer_" + std::to_string(s.layers.size()) + "_rows_" + std::to_string(count));
            host.has_graphs = true;
            auto * source = ggml_view_1d(ctx.ggml, host.source, rows, 0);
            auto * slots = ggml_view_2d(ctx.ggml, host.slots, 1, rows, host.slots->nb[1], host.slot_layout.offsets[bucket++]);
            auto * input = ggml_reshape_3d(ctx.ggml, ggml_get_rows(ctx.ggml, host.hidden, source),
                s.config.input_size, 1, rows);
            auto * output = build_experts(ctx, input, slots, weights, math);
            output = ggml_reshape_2d(ctx.ggml, output, s.config.hidden_size, rows);
            ggml_set_output(output);
            ggml_build_forward_expand(host.graphs.graph(id), output);
            core::validate_backend_graph_supported(backend, host.graphs.graph(id), "staged CPU MoE experts");
            layer.cpu_graphs.push_back(id);
            layer.cpu_outputs.push_back(output);
        }
        s.layers.push_back(std::move(layer));
        return s.layers.size() - 1;
    }
    if (backend != s.execution.backend())
        throw std::invalid_argument("staged MoE cache backend does not match execution");
    if (s.hybrid_enabled && !s.hybrid) {
        s.hybrid = std::make_unique<Impl::Hybrid>(*s.cpu_execution, s.config, s.requirements.phase_times);
        for (auto count : s.hybrid->buckets) {
            auto * graph = ggml_new_graph_custom(s.context.get(), 128, false);
            auto * values = ggml_view_2d(s.context.get(), s.cpu_output, s.config.hidden_size, count, s.cpu_output->nb[1], 0);
            auto * targets = ggml_view_1d(s.context.get(), s.cpu_target, count, 0);
            ggml_build_forward_expand(graph, ggml_set_rows(s.context.get(), s.outputs, values, targets));
            s.hybrid->scatter.push_back(graph);
            s.workspace->add(graph);
        }
    }
    if (placement) {
        layer.placement = placement;
        layer.placement->prepare_program(s.config.identity);
        s.placement_destinations.resize(static_cast<size_t>(s.config.experts));
        layer.decision.gpu.reserve(static_cast<size_t>(s.config.experts));
        layer.decision.cpu.reserve(static_cast<size_t>(s.config.experts));
    }
    core::ModuleBuildContext ctx{s.context.get(), "framework.moe.staged", s.execution.backend_type()};
    if (layer.direct) {
        auto * graph = ggml_new_graph_custom(ctx.ggml, 128, false);
        auto * input = ggml_reshape_3d(ctx.ggml, s.hidden, s.config.input_size, 1, s.config.tokens);
        auto * output = build_experts(ctx, input, s.ids, weights, math);
        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, output, s.result));
        core::validate_backend_graph_supported(backend, graph, "staged MoE experts");
        layer.graphs.push_back(graph);
        s.workspace->add(graph);
        s.layers.push_back(std::move(layer));
        return s.layers.size() - 1;
    }
    size_t bucket = 0;
    for (auto count : s.config.rows_per_expert) {
        const auto rows = count * s.config.cache_slots;
        auto * graph = ggml_new_graph_custom(ctx.ggml, 128, false);
        auto * source = ggml_view_1d(ctx.ggml, s.source, rows, 0);
        auto * target = ggml_view_1d(ctx.ggml, s.target, rows, 0);
        auto * slots = ggml_view_2d(ctx.ggml, s.slots, 1, rows, s.slots->nb[1], s.slot_layout.offsets[bucket++]);
        auto * input = ggml_reshape_3d(ctx.ggml, ggml_get_rows(ctx.ggml, s.hidden, source), s.config.input_size, 1, rows);
        auto * output = build_experts(ctx, input, slots, weights, math);
        ggml_build_forward_expand(graph, ggml_set_rows(ctx.ggml, s.outputs,
            ggml_reshape_2d(ctx.ggml, output, s.config.hidden_size, rows), target));
        core::validate_backend_graph_supported(s.execution.backend(), graph, "staged MoE experts");
        layer.graphs.push_back(graph);
        s.workspace->add(graph);
    }
    if (s.hybrid) {
        const ExpertWeights host_weights{store->host_tensor(group, weights.up).tensor,
            weights.gate ? store->host_tensor(group, weights.gate).tensor : nullptr,
            store->host_tensor(group, weights.down).tensor};
        auto & host = *s.hybrid;
        auto & ctx = host.graphs.build_context();
        for (auto count : host.buckets) {
            const auto id = host.graphs.graph_count() == 1 && ggml_graph_n_nodes(host.graphs.graph()) == 0 ? size_t{0}
                : host.graphs.add_graph("layer_" + std::to_string(s.layers.size()) + "_rows_" + std::to_string(count));
            auto * source = ggml_view_1d(ctx.ggml, host.source, count, 0);
            auto * ids = ggml_view_2d(ctx.ggml, host.ids, 1, count, host.ids->nb[1], 0);
            auto * input = ggml_reshape_3d(ctx.ggml, ggml_get_rows(ctx.ggml, host.hidden, source),
                s.config.input_size, 1, count);
            auto * output = build_experts(ctx, input, ids, host_weights, math);
            ggml_set_output(output);
            ggml_build_forward_expand(host.graphs.graph(id), output);
            core::validate_backend_graph_supported(host.execution.backend(), host.graphs.graph(id), "hybrid MoE CPU experts");
            layer.cpu_graphs.push_back(id);
            layer.cpu_outputs.push_back(output);
        }
    }
    s.layers.push_back(std::move(layer));
    return s.layers.size() - 1;
}

void ExpertProgram::prepare() {
    auto & s = *impl_;
    if (s.prepared || s.layers.empty()) throw std::logic_error("staged MoE preparation requires unprepared layers");
    for (auto & entry : s.layers) for (auto * graph : entry.graphs) {
        auto plan = std::make_unique<core::HostGraphPlan>();
        core::prepare_host_graph_plan(s.execution, graph, *plan);
        entry.plans.push_back(std::move(plan));
    }
    if (s.cpu_stage) s.cpu_stage->graphs.allocate();
    if (s.hybrid) {
        s.hybrid->graphs.allocate();
        for (auto * graph : s.hybrid->scatter) {
            auto plan = std::make_unique<core::HostGraphPlan>();
            core::prepare_host_graph_plan(s.execution, graph, *plan);
            s.hybrid->scatter_plans.push_back(std::move(plan));
        }
        s.hybrid->start();
    }
    s.prepared = true;
}

void ExpertProgram::build_inputs(core::ModuleBuildContext & ctx, ggml_cgraph * graph,
    ggml_tensor * hidden, const Routing & routing) const {
    auto & s = *impl_;
    if (!ctx.elide_identity_reshapes || !ggml_are_same_shape(hidden, s.hidden))
        hidden = ggml_reshape_2d(ctx.ggml, hidden, s.config.input_size, s.config.tokens);
    // Materialize mixing weights before exposing IDs, preserving the existing
    // routing graph order and backend top-k fusion opportunities.
    for (const auto & pair : {std::make_pair(hidden, s.hidden), std::make_pair(routing.weights, s.weights), std::make_pair(routing.ids, s.ids)})
        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, pair.first, pair.second));
}

ggml_tensor * ExpertProgram::hidden_input() const { return impl_->hidden; }
ggml_tensor * ExpertProgram::output() const { return impl_->result; }
ggml_tensor * ExpertProgram::routing_weights() const { return impl_->weights; }
bool ExpertProgram::needs_primary_workspace() const {
    for (const auto & layer : impl_->layers) if (!layer.graphs.empty()) return true;
    return false;
}

uint64_t ExpertProgram::run(size_t layer) {
    auto & s = *impl_;
    if (!s.prepared) throw std::logic_error("staged MoE graphs must be prepared before execution");
    auto & binding = s.layers.at(layer);
    ggml_backend_tensor_get(s.ids, s.logical_ids.data(), 0, s.logical_ids.size() * sizeof(int32_t));
    return s.run(binding);
}

uint64_t ExpertProgram::run(size_t layer, const std::vector<int32_t> & ids) {
    auto & s = *impl_;
    if (!s.prepared || ids.size() != s.logical_ids.size())
        throw std::logic_error("hybrid MoE requires prepared graphs and matching routing IDs");
    std::copy(ids.begin(), ids.end(), s.logical_ids.begin());
    return s.run(s.layers.at(layer));
}

uint64_t ExpertProgram::Impl::run(Layer & binding) {
    auto & s = *this;
    if (s.requirements.cache_accesses) std::fill(s.completed.begin(), s.completed.end(), uint8_t{0});
    const auto cpu_before = s.stats.cpu_assignments;
    const auto gpu_before = s.stats.gpu_assignments;
    const auto cpu_ms_before = s.stats.cpu_ms;
    const auto gpu_ms_before = s.stats.gpu_ms;
    const auto storage_before = binding.store && s.requirements.completed_work
        ? binding.store->statistics() : ExpertStorage::Statistics{};
    uint64_t rows;
    try {
        if (binding.direct) rows = s.run_direct(binding);
        else {
            std::fill(s.assignment_counts.begin(), s.assignment_counts.end(), size_t{0});
            s.policy_experts.clear();
            for (auto id : s.logical_ids) {
                auto & count = s.assignment_counts.at(static_cast<size_t>(id));
                if (count++ == 0) s.policy_experts.push_back(id);
            }
            s.assignment_offsets[0] = 0;
            for (size_t expert = 0; expert < s.assignment_counts.size(); ++expert) {
                s.assignment_cursor[expert] = s.assignment_offsets[expert];
                s.assignment_offsets[expert + 1] = s.assignment_offsets[expert] + s.assignment_counts[expert];
            }
            for (size_t pair = 0; pair < s.logical_ids.size(); ++pair)
                s.assignment_positions[s.assignment_cursor[static_cast<size_t>(s.logical_ids[pair])]++] = static_cast<int32_t>(pair);
            auto & experts = s.experts;
            experts.clear();
            for (size_t expert = 0; expert < s.assignment_counts.size(); ++expert)
                if (s.assignment_counts[expert]) experts.push_back(static_cast<int32_t>(expert));
            const auto & resident = binding.store->cached_experts(binding.group);
            std::sort(experts.begin(), experts.end(), [&](int32_t a, int32_t b) {
                const bool a_hit = std::find(resident.begin(), resident.end(), a) != resident.end();
                const bool b_hit = std::find(resident.begin(), resident.end(), b) != resident.end();
                if (a_hit != b_hit) return a_hit;
                const auto x = s.assignment_counts[static_cast<size_t>(a)], y = s.assignment_counts[static_cast<size_t>(b)];
                return x != y ? x > y : a < b;
            });
            if (binding.cpu_staged) rows = s.run_cpu_staged(binding, experts);
            else if (binding.place_experts) {
                binding.placement->select({s.policy_experts, s.assignment_counts, binding.store->cached_experts(binding.group),
                    binding.store->cache_epoch(), binding.store->expert_bytes(binding.group), s.config.identity}, binding.decision);
                std::fill(s.placement_destinations.begin(), s.placement_destinations.end(), uint8_t{0});
                size_t decisions = 0;
                for (const auto * destination : {&binding.decision.gpu, &binding.decision.cpu}) {
                    for (auto expert : *destination) {
                        if (expert < 0 || static_cast<size_t>(expert) >= s.assignment_counts.size() ||
                            !s.assignment_counts[static_cast<size_t>(expert)] || s.placement_destinations[static_cast<size_t>(expert)]++)
                            throw std::logic_error("MoE placement must partition the routed logical experts");
                        if (destination == &binding.decision.cpu && !s.hybrid)
                            throw std::logic_error("MoE placement selected CPU work without declaring CPU resources");
                        if (destination == &binding.decision.cpu && std::find(resident.begin(), resident.end(), expert) != resident.end())
                            throw std::logic_error("MoE placement cannot move fixed GPU cache hits to CPU");
                        ++decisions;
                    }
                }
                if (decisions != s.policy_experts.size()) throw std::logic_error("MoE placement omitted routed experts");
                rows = s.hybrid ? s.run_hybrid(binding, binding.decision) : s.run_gpu(binding, binding.decision.gpu);
            } else rows = s.run_gpu(binding, experts);
        }
    } catch (...) {
        ggml_backend_synchronize(s.execution.backend());
        if (s.cpu_execution) ggml_backend_synchronize(s.cpu_execution->backend());
        if (s.requirements.cache_accesses)
            for (size_t i = 0; i < s.logical_ids.size(); ++i)
                if (s.completed[i]) binding.store->accessed(binding.group, s.logical_ids[i], 1);
        const PolicyNotification failed{PolicyEvent::ExecutionFailed, s.config.identity,
            binding.store ? binding.store->cache_epoch() : uint64_t{0},
            static_cast<size_t>(s.stats.cpu_assignments - cpu_before),
            static_cast<size_t>(s.stats.gpu_assignments - gpu_before)};
        if (binding.placement) binding.placement->event(failed);
        if (binding.store) binding.store->event(binding.group, failed);
        throw;
    }
    if (binding.placement && s.requirements.completed_work) {
        const auto storage_after = binding.store ? binding.store->statistics() : ExpertStorage::Statistics{};
        binding.placement->observe({static_cast<size_t>(s.stats.cpu_assignments - cpu_before),
            static_cast<size_t>(s.stats.gpu_assignments - gpu_before),
            binding.store ? static_cast<size_t>(storage_after.expert_loads - storage_before.expert_loads) * binding.store->expert_bytes(binding.group) : size_t{0},
            s.requirements.phase_times ? s.stats.cpu_ms - cpu_ms_before : 0,
            s.requirements.phase_times ? s.stats.gpu_ms - gpu_ms_before : 0,
            s.requirements.phase_times ? storage_after.upload_ms - storage_before.upload_ms : 0,
            s.config.identity, s.requirements.phase_times,
            binding.store ? binding.store->cache_epoch() : uint64_t{0}});
    }
    if (s.requirements.cache_accesses)
        for (auto expert : s.logical_ids) binding.store->accessed(binding.group, expert, 1);
    return rows;
}

uint64_t ExpertProgram::Impl::run_direct(Layer & binding) {
    const auto & mapped = binding.store ? binding.store->stage(binding.group, logical_ids) : logical_ids;
    if (binding.cpu_staged) {
        auto & host = *cpu_stage;
        ggml_backend_tensor_get(hidden, host.hidden->data, 0, ggml_nbytes(hidden));
        ggml_backend_tensor_set(host.source, mapped.data(), 0, mapped.size() * sizeof(int32_t));
        host.graphs.compute(binding.cpu_graphs.front());
        const auto * output = binding.cpu_outputs.front();
        ggml_backend_tensor_get(output, host.batch_output.data(), 0, ggml_nbytes(output));
        ggml_backend_tensor_set(result, host.batch_output.data(), 0, ggml_nbytes(output));
        stats.cpu_assignments += logical_ids.size();
    } else {
        ggml_backend_tensor_set(ids, mapped.data(), 0, mapped.size() * sizeof(int32_t));
        if (core::compute_graph(execution, binding.graphs.front(), *binding.plans.front(), "framework.moe.staged") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("staged MoE expert execution failed");
        if (core::is_host_backend(execution.backend())) stats.cpu_assignments += logical_ids.size();
        else stats.gpu_assignments += logical_ids.size();
    }
    if (requirements.cache_accesses) std::fill(completed.begin(), completed.end(), uint8_t{1});
    return logical_ids.size();
}

uint64_t ExpertProgram::Impl::run_cpu_staged(Layer & binding, const std::vector<int32_t> & experts) {
    auto & s = *this;
    auto & host = *s.cpu_stage;
    ggml_backend_tensor_get(s.hidden, host.hidden->data, 0, ggml_nbytes(s.hidden));
    uint64_t work_rows = 0;
    for (size_t first = 0; first < experts.size(); first += s.config.cache_slots) {
        auto & group = s.group;
        group.assign(experts.begin() + first, experts.begin() + std::min(experts.size(), first + static_cast<size_t>(s.config.cache_slots)));
        const auto & mapped = binding.store->stage(binding.group, group);
        size_t maximum = 0;
        for (auto expert : group) maximum = std::max(maximum, s.assignment_counts[static_cast<size_t>(expert)]);
        for (size_t begin = 0; begin < maximum; begin += s.config.rows_per_expert.back()) {
            const auto needed = std::min(maximum - begin, static_cast<size_t>(s.config.rows_per_expert.back()));
            const auto bucket = std::lower_bound(s.config.rows_per_expert.begin(), s.config.rows_per_expert.end(), needed);
            const auto count = *bucket;
            const auto rows = static_cast<size_t>(count * s.config.cache_slots);
            std::fill_n(s.source_rows.begin(), rows, 0);
            std::visit([&](auto & targets) {
                using Index = typename std::decay_t<decltype(targets)>::value_type;
                for (size_t row = 0; row < rows; ++row)
                    targets[row] = static_cast<Index>(s.config.selected * s.config.tokens + row);
                for (size_t e = 0; e < group.size(); ++e) {
                    const auto expert = static_cast<size_t>(group[e]);
                    const auto end = std::min(s.assignment_counts[expert], begin + static_cast<size_t>(count));
                    for (size_t row = begin; row < end; ++row) {
                        const auto assignment = s.assignment_positions[s.assignment_offsets[expert] + row];
                        const auto execution_row = mapped[e] * count + row - begin;
                        s.source_rows[execution_row] = static_cast<int32_t>(assignment / s.config.selected);
                        targets[execution_row] = static_cast<Index>(assignment);
                    }
                }
                ggml_backend_tensor_set(host.source, s.source_rows.data(), 0, rows * sizeof(int32_t));
            }, s.target_rows);
            const auto index = static_cast<size_t>(bucket - s.config.rows_per_expert.begin());
            const auto started = s.requirements.phase_times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            host.graphs.compute(binding.cpu_graphs[index]);
            if (s.requirements.phase_times) s.stats.cpu_ms += debug::elapsed_ms(started);
            ggml_backend_tensor_get(binding.cpu_outputs[index], host.batch_output.data(), 0,
                rows * static_cast<size_t>(s.config.hidden_size) * sizeof(float));
            std::visit([&](const auto & targets) {
                const auto width = static_cast<size_t>(s.config.hidden_size);
                for (size_t row = 0; row < rows; ++row) {
                    const auto assignment = static_cast<size_t>(targets[row]);
                    if (assignment < s.logical_ids.size()) {
                        std::copy_n(host.batch_output.data() + row * width, width,
                            host.ordered_output.data() + assignment * width);
                        ++s.stats.cpu_assignments;
                        if (s.requirements.cache_accesses) s.completed[assignment] = 1;
                    }
                }
            }, s.target_rows);
            work_rows += rows;
        }
    }
    ggml_backend_tensor_set(s.outputs, host.ordered_output.data(), 0, host.ordered_output.size() * sizeof(float));
    return work_rows;
}

uint64_t ExpertProgram::Impl::run_gpu(Layer & binding, const std::vector<int32_t> & experts) {
    auto & s = *this;
    uint64_t work_rows = 0;
    for (size_t first = 0; first < experts.size(); first += s.config.cache_slots) {
        auto & group = s.group;
        group.assign(experts.begin() + first, experts.begin() + std::min(experts.size(), first + static_cast<size_t>(s.config.cache_slots)));
        const auto & mapped = binding.store->stage(binding.group, group);
        size_t maximum = 0;
        for (auto expert : group) maximum = std::max(maximum, s.assignment_counts[static_cast<size_t>(expert)]);
        for (size_t begin = 0; begin < maximum; begin += s.config.rows_per_expert.back()) {
            const auto needed = std::min(maximum - begin, static_cast<size_t>(s.config.rows_per_expert.back()));
            const auto bucket = std::lower_bound(s.config.rows_per_expert.begin(), s.config.rows_per_expert.end(), needed);
            const auto count = *bucket;
            const auto rows = static_cast<size_t>(count * s.config.cache_slots);
            std::fill_n(s.source_rows.begin(), rows, 0);
            std::visit([&](auto & targets) {
                using Index = typename std::decay_t<decltype(targets)>::value_type;
                for (size_t row = 0; row < rows; ++row)
                    targets[row] = static_cast<Index>(s.config.selected * s.config.tokens + row);
                for (size_t e = 0; e < group.size(); ++e) {
                    const auto expert = static_cast<size_t>(group[e]);
                    const auto end = std::min(s.assignment_counts[expert], begin + static_cast<size_t>(count));
                    for (size_t row = begin; row < end; ++row) {
                        const auto assignment = s.assignment_positions[s.assignment_offsets[expert] + row];
                        const auto execution_row = mapped[e] * count + row - begin;
                        s.source_rows[execution_row] = static_cast<int32_t>(assignment / s.config.selected);
                        targets[execution_row] = static_cast<Index>(assignment);
                    }
                }
                ggml_backend_tensor_set(s.source, s.source_rows.data(), 0, rows * sizeof(int32_t));
                ggml_backend_tensor_set(s.target, targets.data(), 0, rows * sizeof(Index));
            }, s.target_rows);
            const auto index = static_cast<size_t>(bucket - s.config.rows_per_expert.begin());
            auto * graph = binding.graphs[index];
            const auto started = s.requirements.phase_times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (core::compute_graph(s.execution, graph, *binding.plans[index], "framework.moe.staged") != GGML_STATUS_SUCCESS)
                throw std::runtime_error("staged MoE expert execution failed");
            const bool host_backend = core::is_host_backend(s.execution.backend());
            if (s.requirements.phase_times) {
                if (host_backend) s.stats.cpu_ms += debug::elapsed_ms(started);
                else s.stats.gpu_ms += debug::elapsed_ms(started);
            }
            for (auto id : group) {
                const auto expert = static_cast<size_t>(id);
                const auto end = std::min(s.assignment_counts[expert], begin + static_cast<size_t>(count));
                if (end <= begin) continue;
                if (host_backend) s.stats.cpu_assignments += end - begin;
                else s.stats.gpu_assignments += end - begin;
                if (s.requirements.cache_accesses)
                    for (size_t row = begin; row < end; ++row)
                        s.completed[static_cast<size_t>(s.assignment_positions[s.assignment_offsets[expert] + row])] = 1;
            }
            work_rows += rows;
        }
    }
    return work_rows;
}

uint64_t ExpertProgram::Impl::run_hybrid(Layer & binding, const PlacementDecision & placement) {
    auto & host = *hybrid;
    size_t cpu_rows = 0;
    for (auto expert : placement.cpu) cpu_rows += assignment_counts[static_cast<size_t>(expert)];
    if (cpu_rows == 0) return run_gpu(binding, placement.gpu);
    const auto found = std::lower_bound(host.buckets.begin(), host.buckets.end(), static_cast<int64_t>(cpu_rows));
    const auto bucket = static_cast<size_t>(found - host.buckets.begin());
    const auto rows = static_cast<size_t>(*found);
    std::fill_n(host.source_rows.begin(), rows, 0);
    std::fill_n(host.expert_ids.begin(), rows, 0);
    std::visit([&](auto & targets) {
        using Index = typename std::decay_t<decltype(targets)>::value_type;
        for (size_t row = 0; row < rows; ++row) targets[row] = static_cast<Index>(config.selected * config.tokens + row);
        size_t row = 0;
        for (auto expert : placement.cpu) for (size_t offset = assignment_offsets[static_cast<size_t>(expert)];
            offset < assignment_offsets[static_cast<size_t>(expert) + 1]; ++offset) {
            const auto assignment = assignment_positions[offset];
            host.source_rows[row] = static_cast<int32_t>(assignment / config.selected);
            host.expert_ids[row] = expert;
            targets[row++] = static_cast<Index>(assignment);
        }
        ggml_backend_tensor_set(cpu_target, targets.data(), 0, rows * sizeof(Index));
    }, host.target_rows);
    // Only activations cross the device boundary; expert payloads stay native
    // in the mapped/pinned host bank. CPU work overlaps uploads/GPU execution.
    ggml_backend_tensor_get(hidden, host.hidden->data, 0, ggml_nbytes(hidden));
    ggml_backend_tensor_set(host.source, host.source_rows.data(), 0, rows * sizeof(int32_t));
    ggml_backend_tensor_set(host.ids, host.expert_ids.data(), 0, rows * sizeof(int32_t));
    host.launch(binding, bucket);
    uint64_t work_rows;
    try { work_rows = run_gpu(binding, placement.gpu); }
    catch (...) {
        host.wait();
        stats.cpu_assignments += cpu_rows;
        stats.cpu_ms += host.compute_ms;
        if (requirements.cache_accesses)
            for (auto expert : placement.cpu)
                for (size_t offset = assignment_offsets[static_cast<size_t>(expert)]; offset < assignment_offsets[static_cast<size_t>(expert) + 1]; ++offset)
                    completed[static_cast<size_t>(assignment_positions[offset])] = 1;
        throw;
    }
    host.wait();
    stats.cpu_assignments += cpu_rows;
    stats.cpu_ms += host.compute_ms;
    if (requirements.cache_accesses)
        for (auto expert : placement.cpu)
            for (size_t offset = assignment_offsets[static_cast<size_t>(expert)]; offset < assignment_offsets[static_cast<size_t>(expert) + 1]; ++offset)
                completed[static_cast<size_t>(assignment_positions[offset])] = 1;
    const auto * output = binding.cpu_outputs[bucket];
    ggml_backend_tensor_set_async(execution.backend(), cpu_output, output->data, 0, ggml_nbytes(output));
    auto * scatter = host.scatter[bucket];
    if (core::compute_graph(execution, scatter, *host.scatter_plans[bucket], "framework.moe.cpu_merge") != GGML_STATUS_SUCCESS)
        throw std::runtime_error("hybrid MoE output scatter failed");
    // The synchronous merge also completes the upload before the next CPU
    // graph can reuse its scratch. Mixing remains in original logical order.
    return work_rows + rows;
}

} // namespace engine::moe::detail
