#include "engine/framework/moe_engine/moe_engine.h"
#include "engine/framework/moe_engine/moe_prepared_graph.h"
#include "moe_engine_internal.h"
#include "engine/framework/core/backend.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace engine::moe {
class MoeEngine::Impl {
public:
    struct Layer {
        LayerBindings bindings;
        LayerMath math;
        detail::ExpertWeights weights;
        ggml_backend_t backend = nullptr;
        size_t cache_group = 0;
        std::unique_ptr<PlacementPolicy> placement;
    };
    struct Program {
        CallShape shape;
        size_t experts = 0, selected = 0;
        std::shared_ptr<core::GraphWorkspace> workspace;
        std::unique_ptr<detail::ExpertProgram> execution;
        bool owns_workspace = false, prepared = false;
    };
    struct Call {
        size_t layer = 0, execution_layer = 0;
        uint64_t generation = 0;
        std::shared_ptr<Program> program;
        bool native_routing = false;
    };
    core::ExecutionContext & execution;
    core::BackendWeightStore & primary;
    const assets::TensorSource & source;
    EngineConfig config;
    PreparedPolicy policy;
    std::string label;
    size_t metadata_bytes;
    // Destruction order: programs, storage/weights, then auxiliary backend.
    std::unique_ptr<core::ExecutionContext> cpu;
    std::unique_ptr<core::BackendWeightStore> cpu_weights;
    std::unique_ptr<detail::ExpertStorage> storage;
    std::vector<Layer> layers;
    std::vector<std::weak_ptr<Program>> programs;
    std::vector<std::unique_ptr<Call>> calls;
    std::mutex mutex;
    bool finalized = false, active = false;
    uint64_t generation = 0;

    Impl(core::ExecutionContext & e, core::BackendWeightStore & weights,
        const assets::TensorSource & tensors, EngineConfig c, std::string name, size_t bytes)
        : execution(e), primary(weights), source(tensors), config(std::move(c)),
          policy(prepare_policy(config.policies)), label(std::move(name)), metadata_bytes(bytes) {
        if (config.cache_slots <= 0 || config.read_threads < 1 || config.read_threads > 64)
            throw std::invalid_argument("invalid MoE cache capacity or read threads");
        if (policy.active_placement && config.residency != Residency::Staged)
            throw std::invalid_argument("active MoE placement requires staged expert storage");
        if (config.policies.eviction.name != "default" && config.residency != Residency::Staged)
            throw std::invalid_argument("MoE eviction policy requires staged expert storage");
        auto ordinals = config.cpu_layers;
        std::sort(ordinals.begin(), ordinals.end());
        if (std::adjacent_find(ordinals.begin(), ordinals.end()) != ordinals.end())
            throw std::invalid_argument("duplicate MoE CPU layer ordinal");
        if ((!ordinals.empty() || policy.requirements.cpu_execution) && !core::is_host_backend(execution.backend())) {
            auto backend = core::BackendConfig{};
            backend.threads = execution.config().threads;
            cpu = std::make_unique<core::ExecutionContext>(backend);
        }
        if (config.residency == Residency::Staged)
            storage = std::make_unique<detail::ExpertStorage>(source.source_path(), config.pin_host_weights,
                static_cast<size_t>(config.read_threads), execution.backend(), policy.eviction);
    }
    core::ExecutionContext * cpu_context() {
        return cpu ? cpu.get() : (core::is_host_backend(execution.backend()) ? &execution : nullptr);
    }
    core::ExecutionContext & require_cpu_context() {
        if (core::is_host_backend(execution.backend())) return execution;
        if (!cpu) {
            core::BackendConfig backend;
            backend.threads = execution.config().threads;
            cpu = std::make_unique<core::ExecutionContext>(backend);
        }
        return *cpu;
    }
    void idle() const { if (active) throw std::logic_error("cannot mutate MoE preparation during execution"); }
    Layer & layer(LayerId id) {
        if (id.owner != this || id.index >= layers.size()) throw std::invalid_argument("invalid MoE layer handle");
        return layers[id.index];
    }
    Call & call(CallId id) {
        if (id.owner != this || id.index >= calls.size() || !calls[id.index] || calls[id.index]->generation != id.generation)
            throw std::invalid_argument("invalid or retired MoE call handle");
        return *calls[id.index];
    }
    CallId retain_call(std::unique_ptr<Call> call) {
        call->generation = ++generation;
        size_t index = 0;
        while (index < calls.size() && calls[index]) ++index;
        if (index == calls.size()) calls.emplace_back();
        const auto version = call->generation;
        calls[index] = std::move(call);
        return {index, this, version};
    }
};

MoeEngine::MoeEngine(core::ExecutionContext & execution, core::BackendWeightStore & primary,
    const assets::TensorSource & source, EngineConfig config, std::string label, size_t bytes)
    : impl_(std::make_unique<Impl>(execution, primary, source, std::move(config), std::move(label), bytes)) {}
MoeEngine::~MoeEngine() {
    impl_->calls.clear();
    impl_->programs.clear();
}
const EngineConfig & MoeEngine::config() const noexcept { return impl_->config; }
bool MoeEngine::staged() const noexcept { return impl_->storage != nullptr; }
core::ExecutionContext & MoeEngine::execution() const { return impl_->execution; }

LayerId MoeEngine::register_layer(LayerBindings bindings, LayerMath math) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto & s = *impl_;
    s.idle();
    if (s.finalized) throw std::logic_error("MoE weight registration is finalized");
    if (bindings.experts.size() != 2 && bindings.experts.size() != 3)
        throw std::invalid_argument("MoE layer requires up/down or up/gate/down expert projections");
    const auto experts = math.routing.experts;
    if (experts <= 0 || math.routing.selected <= 0 || math.routing.selected > experts ||
        (s.storage && s.config.cache_slots > experts)) throw std::invalid_argument("invalid MoE layer dimensions");
    if (math.experts.fused_swiglu && (math.experts.activation != ExpertActivation::SwiGlu || math.experts.round_activation))
        throw std::invalid_argument("fused MoE SwiGLU cannot remove activation rounding");
    unsigned roles = 0;
    for (const auto & tensor : bindings.experts) {
        const auto role = 1u << static_cast<unsigned>(tensor.role);
        if (roles & role) throw std::invalid_argument("duplicate MoE expert projection role");
        roles |= role;
    }
    const auto required_roles = (1u << static_cast<unsigned>(ProjectionRole::Up)) | (1u << static_cast<unsigned>(ProjectionRole::Down));
    if ((roles & required_roles) != required_roles || (math.experts.activation == ExpertActivation::SwiGlu && !(roles & (1u << static_cast<unsigned>(ProjectionRole::Gate)))))
        throw std::invalid_argument("missing MoE expert projection role");
    for (const auto & tensor : bindings.experts) {
        if (tensor.shape.size() != 3 || tensor.shape[0] != experts || s.source.require_metadata(tensor.name).shape != tensor.shape)
            throw std::invalid_argument("MoE expert bank shape mismatch: " + tensor.name);
        if (s.storage && tensor.storage != assets::TensorStorageType::Native)
            throw std::invalid_argument("staged MoE experts require native storage");
    }
    Impl::Layer layer;
    layer.math = math;
    layer.bindings = std::move(bindings);
    const size_t ordinal = s.layers.size();
    const bool cpu = std::find(s.config.cpu_layers.begin(), s.config.cpu_layers.end(), ordinal) != s.config.cpu_layers.end();
    layer.backend = cpu ? s.cpu_context()->backend() : s.execution.backend();
    s.layers.reserve(ordinal + 1);
    std::vector<ggml_tensor *> tensors;
    if (s.storage) {
        std::vector<std::string> names;
        for (const auto & binding : layer.bindings.experts) names.push_back(binding.disk_name);
        layer.cache_group = s.storage->add_group(names, layer.backend, s.config.cache_slots);
        for (size_t i = 0; i < names.size(); ++i) tensors.push_back(s.storage->tensor(layer.cache_group, i).tensor);
    } else {
        auto * store = &s.primary;
        if (cpu && s.cpu) {
            if (!s.cpu_weights) s.cpu_weights = std::make_unique<core::BackendWeightStore>(s.cpu->backend(), core::BackendType::Cpu, s.label, s.metadata_bytes);
            store = s.cpu_weights.get();
        }
        for (const auto & binding : layer.bindings.experts)
            tensors.push_back(store->load_tensor(s.source, binding.name, binding.storage, binding.shape).tensor);
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_tensor ** target = nullptr;
        switch (layer.bindings.experts[i].role) {
            case ProjectionRole::Up: target = &layer.weights.up; break;
            case ProjectionRole::Gate: target = &layer.weights.gate; break;
            case ProjectionRole::Down: target = &layer.weights.down; break;
        }
        if (*target) throw std::invalid_argument("duplicate MoE expert projection role");
        *target = tensors[i];
    }
    if (!layer.weights.up || !layer.weights.down || (math.experts.activation == ExpertActivation::SwiGlu && !layer.weights.gate))
        throw std::invalid_argument("missing MoE expert projection role");
    if (s.policy.active_placement) layer.placement = s.policy.placement(static_cast<size_t>(experts));
    s.layers.push_back(std::move(layer));
    return {ordinal, impl_.get()};
}
void MoeEngine::finalize_weights() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->idle();
    if (impl_->finalized) throw std::logic_error("MoE weights already finalized");
    for (auto ordinal : impl_->config.cpu_layers)
        if (ordinal >= impl_->layers.size()) throw std::invalid_argument("MoE CPU layer ordinal was not registered");
    if (impl_->cpu_weights) impl_->cpu_weights->upload();
    impl_->finalized = true;
}
ExpertStatistics MoeEngine::statistics() const noexcept {
    if (!impl_->storage) return {};
    const auto s = impl_->storage->statistics();
    return {s.bytes_read, s.expert_loads, s.cache_hits, s.buffer_bytes, s.host_buffer_bytes, s.read_ms, s.upload_ms, s.stage_ms};
}
void MoeEngine::begin_generation(bool invalidate_weights) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->idle();
    const auto before = impl_->storage ? impl_->storage->cache_epoch() : uint64_t{0};
    for (auto & layer : impl_->layers) if (layer.placement)
        layer.placement->event({PolicyEvent::GenerationBegin, 0, before});
    if (impl_->storage) impl_->storage->event({PolicyEvent::GenerationBegin, 0, before});
    if (invalidate_weights && impl_->storage) {
        impl_->storage->invalidate_cache();
        const auto epoch = impl_->storage->cache_epoch();
        for (auto & layer : impl_->layers) if (layer.placement)
            layer.placement->event({PolicyEvent::CacheInvalidated, 0, epoch});
    }
}

CallId MoeEngine::add_call(LayerId id, CallShape shape, std::shared_ptr<core::GraphWorkspace> workspace) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto & s = *impl_; s.idle();
    auto & layer = s.layer(id);
    if (!s.finalized) throw std::logic_error("prepared MoE calls require finalized weights");
    if (shape.tokens <= 0 || shape.minimum_rows <= 0 || shape.maximum_rows <= 0) throw std::invalid_argument("invalid MoE call shape");
    if (!shape.input_width) shape.input_width = layer.weights.up->ne[0];
    if (!shape.output_width) shape.output_width = layer.weights.down->ne[1];
    std::shared_ptr<Impl::Program> program;
    for (auto it = s.programs.begin(); it != s.programs.end();) {
        auto candidate = it->lock();
        if (!candidate) { it = s.programs.erase(it); continue; }
        const auto & c = candidate->shape;
        if (!candidate->prepared && candidate->workspace == workspace && workspace &&
            c.tokens == shape.tokens && c.input_width == shape.input_width && c.output_width == shape.output_width &&
            c.maximum_rows == shape.maximum_rows && c.minimum_rows == shape.minimum_rows && c.lowering == shape.lowering && c.scatter_index_type == shape.scatter_index_type &&
            c.native_single_token == shape.native_single_token &&
            candidate->experts == static_cast<size_t>(layer.math.routing.experts) && candidate->selected == static_cast<size_t>(layer.math.routing.selected)) program = candidate;
        ++it;
    }
    if (!program) {
        program = std::make_shared<Impl::Program>();
        program->shape = shape;
        program->experts = static_cast<size_t>(layer.math.routing.experts);
        program->selected = static_cast<size_t>(layer.math.routing.selected);
        program->owns_workspace = !workspace;
        program->workspace = workspace ? std::move(workspace) : std::make_shared<core::GraphWorkspace>(s.execution.backend());
        detail::ProgramShape config;
        config.identity = ++s.generation;
        config.resident = !s.storage;
        config.native_single_token = shape.native_single_token;
        config.hidden_size = shape.output_width; config.input_size = shape.input_width;
        config.tokens = shape.tokens; config.selected = layer.math.routing.selected;
        config.experts = layer.math.routing.experts;
        config.cache_slots = s.storage ? s.config.cache_slots : layer.math.routing.experts;
        config.scatter_index_type = shape.scatter_index_type;
        int64_t maximum = std::max(shape.minimum_rows, std::min(shape.tokens, shape.maximum_rows));
        int64_t bucket = 1; while (bucket < maximum) bucket *= 2;
        for (int64_t count = shape.minimum_rows; count <= bucket; count *= 2) config.rows_per_expert.push_back(count);
        program->execution = std::make_unique<detail::ExpertProgram>(s.execution, std::move(config), program->workspace, s.cpu_context(), s.policy);
        s.programs.push_back(program);
    }
    auto call = std::make_unique<Impl::Call>();
    call->layer = id.index;
    auto math = layer.math.experts;
    if (shape.lowering == ExpertLowering::FusedSwiGlu) math.fused_swiglu = true;
    if (math.fused_swiglu && math.round_activation) throw std::invalid_argument("fused expert lowering cannot remove activation rounding");
    call->execution_layer = program->execution->add_layer(math, layer.weights, s.storage.get(), layer.cache_group, layer.backend, layer.placement.get());
    call->program = std::move(program);
    return s.retain_call(std::move(call));
}
void MoeEngine::build_inputs(CallId id, MoePreparedGraph & graph, size_t graph_id, ggml_tensor * input, const Routing * external) {
    auto & call = impl_->call(id);
    auto & layer = impl_->layers[call.layer];
    auto & ctx = graph.build_context();
    const auto routing = external ? *external : detail::build_routing(ctx, input, layer.bindings.router.tensor, layer.bindings.selection_bias.tensor, layer.math.routing);
    call.program->execution->build_inputs(ctx, graph.graph(graph_id), input, routing);
    graph.bind_call(*this, id);
}
ggml_tensor * MoeEngine::build_result(CallId id, MoePreparedGraph & graph) {
    auto & call = impl_->call(id);
    graph.bind_call(*this, id);
    return detail::combine(graph.build_context(), call.program->execution->output(), call.program->execution->routing_weights());
}
Routing MoeEngine::route(core::ModuleBuildContext & ctx, LayerId id, ggml_tensor * input) const {
    const auto & layer = impl_->layer(id);
    return detail::build_routing(ctx, input, layer.bindings.router.tensor, layer.bindings.selection_bias.tensor, layer.math.routing);
}
ggml_tensor * MoeEngine::build(MoePreparedGraph & graph, LayerId id, ggml_tensor * input, const Routing * external, ExpertLowering lowering) {
    auto & layer = impl_->layer(id);
    if (!impl_->finalized) throw std::logic_error("MoE graph construction requires finalized weights");
    auto & ctx = graph.build_context();
    const auto routing = external ? *external : detail::build_routing(ctx, input, layer.bindings.router.tensor, layer.bindings.selection_bias.tensor, layer.math.routing);
    auto * rows = ggml_reshape_3d(ctx.ggml, input, input->ne[0], 1, input->ne[1]);
    ggml_tensor * first = nullptr;
    auto math = layer.math.experts;
    if (lowering == ExpertLowering::FusedSwiGlu) {
        if (math.activation != ExpertActivation::SwiGlu || math.round_activation)
            throw std::invalid_argument("fused expert lowering is incompatible with the registered mathematics");
        math.fused_swiglu = true;
        if (!impl_->storage) ggml_build_forward_expand(graph.graph(), routing.weights);
    }
    // Native single-token execution needs only logical-to-cache-slot mapping.
    // Keep its expert operations in the parent graph, preserving its lowering
    // and avoiding separate expert graphs and boundary-copy submissions.
    // Policies requesting completion observations still use explicit execution.
    const auto & requirements = impl_->policy.requirements;
    const bool native_routing = impl_->storage && input->ne[1] == 1 &&
        layer.math.routing.selected <= impl_->config.cache_slots &&
        !impl_->policy.active_placement && !requirements.cache_accesses &&
        !requirements.completed_work && !requirements.phase_times;
    auto * expert_ids = routing.ids;
    if (native_routing) {
        expert_ids = ggml_cont(ctx.ggml, routing.ids);
        ggml_set_output(expert_ids);
        auto call = std::make_unique<Impl::Call>();
        call->layer = id.index;
        call->native_routing = true;
        CallId binding;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->idle();
            binding = impl_->retain_call(std::move(call));
        }
        graph.bind_cache_route(*this, binding, routing.ids, expert_ids, layer.backend);
        if (layer.backend != impl_->execution.backend()) graph.place_nodes({expert_ids}, layer.backend);
    }
    auto * output = detail::build_experts(ctx, rows, expert_ids, layer.weights, math, &first);
    if (impl_->storage && !native_routing) {
        auto * boundary = first->src[1];
        auto call = add_call(id, {input->ne[1], boundary->ne[0], output->ne[0], input->ne[1], GGML_TYPE_I64, 1, lowering, true}, graph.embedded_workspace());
        graph.bind_region(*this, call, boundary, routing.ids, first, output);
    } else if (layer.backend != impl_->execution.backend()) {
        std::vector<ggml_tensor *> nodes;
        const auto visit = [&](const auto & self, ggml_tensor * node) -> void {
            if (!node || node == rows || node == expert_ids || node->op == GGML_OP_NONE || std::find(nodes.begin(), nodes.end(), node) != nodes.end()) return;
            for (auto * source : node->src) self(self, source);
            nodes.push_back(node);
        };
        visit(visit, output);
        graph.require_backend(layer.backend); graph.place_nodes(nodes, layer.backend);
    }
    return detail::combine(ctx, output, routing.weights);
}
bool MoeEngine::needs_primary_workspace(CallId id) const {
    const auto & call = impl_->call(id);
    return !call.native_routing && call.program->execution->needs_primary_workspace();
}
void MoeEngine::prepare(CallId id) {
    std::lock_guard<std::mutex> lock(impl_->mutex); impl_->idle();
    auto & call = impl_->call(id);
    if (call.native_routing) return;
    auto & program = *call.program;
    if (program.prepared) return;
    if (program.owns_workspace && program.execution->needs_primary_workspace()) program.workspace->prepare();
    program.execution->prepare(); program.prepared = true;
}
uint64_t MoeEngine::execute(CallId id, const std::vector<int32_t> * ids) {
    if (!impl_->active) throw std::logic_error("MoE execution requires a graph-owned lease");
    auto & call = impl_->call(id);
    return ids ? call.program->execution->run(call.execution_layer, *ids) : call.program->execution->run(call.execution_layer);
}
const std::vector<int32_t> & MoeEngine::stage_routes(CallId id, const std::vector<int32_t> & ids) {
    if (!impl_->active) throw std::logic_error("MoE routing requires a graph-owned lease");
    const auto & call = impl_->call(id);
    const auto & layer = impl_->layers[call.layer];
    return impl_->storage->stage(layer.cache_group, ids);
}
ggml_backend_t MoeEngine::routing_backend() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->idle();
    return impl_->require_cpu_context().backend();
}
ggml_tensor * MoeEngine::input_port(CallId id) const { return impl_->call(id).program->execution->hidden_input(); }
ggml_tensor * MoeEngine::output_port(CallId id) const { return impl_->call(id).program->execution->output(); }
void MoeEngine::acquire() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->active) throw std::runtime_error("overlapping MoE graph sequences are not supported");
    impl_->active = true;
}
void MoeEngine::drain() {
    ggml_backend_synchronize(impl_->execution.backend());
    if (impl_->cpu) ggml_backend_synchronize(impl_->cpu->backend());
}
void MoeEngine::relinquish() noexcept { std::lock_guard<std::mutex> lock(impl_->mutex); impl_->active = false; }
void MoeEngine::release(CallId id) {
    std::lock_guard<std::mutex> lock(impl_->mutex); impl_->idle(); impl_->call(id); impl_->calls[id.index].reset();
}
void MoeEngine::release_execution() {
    std::lock_guard<std::mutex> lock(impl_->mutex); impl_->idle();
    if (std::any_of(impl_->calls.begin(), impl_->calls.end(), [](const auto & call) { return bool(call); }))
        throw std::logic_error("release graph bindings before releasing MoE programs");
    drain();
    impl_->programs.clear();
    const auto epoch = impl_->storage ? impl_->storage->cache_epoch() : uint64_t{0};
    for (auto & layer : impl_->layers) if (layer.placement)
        layer.placement->event({PolicyEvent::ExecutionReleased, 0, epoch});
    if (impl_->storage) impl_->storage->event({PolicyEvent::ExecutionReleased, 0, epoch});
}
} // namespace engine::moe
