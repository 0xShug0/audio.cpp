#include "engine/framework/moe_engine/moe_prepared_graph.h"
#include "engine/framework/core/backend.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace engine::moe {
class MoePreparedGraph::Scope {
public:
    explicit Scope(MoePreparedGraph & graph) : graph_(graph) {
        if (graph.active_.exchange(true)) throw std::runtime_error("overlapping MoE graph sequences are not supported");
        try {
            for (auto * engine : graph.engines_) { engine->acquire(); ++acquired_; }
        } catch (...) {
            for (size_t i = 0; i < acquired_; ++i) graph.engines_[i]->relinquish();
            graph.active_ = false; throw;
        }
    }
    ~Scope() {
        if (!finished_) {
            graph_.drain();
            for (size_t i = 0; i < acquired_; ++i) graph_.engines_[i]->drain();
            release();
        }
    }
    void finish() {
        // PreparedGraph::compute and expert execution are synchronous. Their
        // return establishes consumer completion; do not add another device
        // synchronization to every successful decode sequence.
        release();
        finished_ = true;
    }
    void release() noexcept {
        for (size_t i = 0; i < acquired_; ++i) graph_.engines_[i]->relinquish();
        graph_.active_ = false;
    }

private:
    MoePreparedGraph & graph_;
    size_t acquired_ = 0;
    bool finished_ = false;
};

// Boundary-copy metadata belongs to the parent integration, not to the engine.
class MoePreparedGraph::Region final : public core::GraphRegionExecutor {
public:
    Region(MoePreparedGraph & parent, MoeEngine & engine, CallId call,
        ggml_tensor * input, ggml_tensor * ids, ggml_tensor * output)
        : parent_(parent), engine_(engine), call_(call), input_(input), ids_(ids), output_(output) {}
    void prepare(ggml_backend_t backend, ggml_cgraph *) override {
        if (backend != engine_.execution().backend())
            throw std::runtime_error("embedded MoE region must retain its primary backend");
        engine_.prepare(call_);
        copies_ = std::make_unique<core::PreparedGraph>(engine_.execution(),
            core::PreparedGraphConfig{"framework.moe.boundary", 128 * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(64, false), 64});
        auto & ctx = copies_->build_context();
        const auto leaf = [&](ggml_tensor * original) {
            auto * tensor = ggml_dup_tensor(ctx.ggml, original);
            if (ggml_backend_tensor_alloc(original->buffer, tensor, original->data) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("cannot bind MoE graph boundary storage");
            return tensor;
        };
        auto * input = leaf(input_);
        auto * result = leaf(output_);
        ggml_build_forward_expand(copies_->graph(), ggml_cpy(ctx.ggml,
            ggml_reshape_2d(ctx.ggml, input, input_->ne[0], ggml_nelements(input_) / input_->ne[0]), engine_.input_port(call_)));
        output_copy_ = copies_->add_graph("result");
        ggml_build_forward_expand(copies_->graph(output_copy_), ggml_cpy(ctx.ggml, engine_.output_port(call_), result));
        copies_->allocate();
        ids_buffer_.resize(static_cast<size_t>(ggml_nelements(ids_)));
    }
    ggml_status compute() override {
        if (!parent_.active_) throw std::logic_error("embedded MoE requires its enclosing graph scope");
        copies_->compute();
        ggml_backend_tensor_get(ids_, ids_buffer_.data(), 0, ids_buffer_.size() * sizeof(int32_t));
        engine_.execute(call_, &ids_buffer_);
        copies_->compute(output_copy_);
        return GGML_STATUS_SUCCESS;
    }
    void reset() noexcept override { copies_.reset(); }
private:
    MoePreparedGraph & parent_;
    MoeEngine & engine_;
    CallId call_;
    ggml_tensor * input_, * ids_, * output_;
    std::unique_ptr<core::PreparedGraph> copies_;
    size_t output_copy_ = 0;
    std::vector<int32_t> ids_buffer_;
};

// Default native expert math remains in the parent graph. This region replaces
// only the ID copy with cache mapping; the enclosing lease protects the weights
// until the parent's expert operations and consumers complete.
class MoePreparedGraph::CacheRoute final : public core::GraphRegionExecutor {
public:
    friend class MoePreparedGraph;
    CacheRoute(MoePreparedGraph & parent, MoeEngine & engine, CallId call,
        ggml_tensor * ids, ggml_tensor * mapped, ggml_backend_t backend)
        : parent_(parent), engine_(engine), call_(call), ids_(ids), mapped_(mapped), backend_(backend),
          logical_ids_(static_cast<size_t>(ggml_nelements(ids))) {}
    void prepare(ggml_backend_t backend, ggml_cgraph *) override {
        if (backend != backend_) throw std::runtime_error("native MoE routing must retain its expert backend");
        source_backend_ = parent_.scheduler_
            ? ggml_backend_sched_get_tensor_backend(parent_.scheduler_, ids_) : parent_.execution().backend();
        if (!source_backend_) throw std::runtime_error("native MoE routing requires the ID producer backend");
    }
    ggml_status compute() override {
        if (!parent_.active_) throw std::logic_error("native MoE routing requires its enclosing graph scope");
        // Scheduled producers may still be queued. Complete their work before
        // the synchronous read, matching the ordinary cache callback contract.
        if (!evaluated_) {
            ggml_backend_synchronize(source_backend_);
            if (backend_ != source_backend_) ggml_backend_synchronize(backend_);
        }
        ggml_backend_tensor_get(evaluated_ ? mapped_ : ids_, logical_ids_.data(), 0, logical_ids_.size() * sizeof(int32_t));
        const auto & mapped = engine_.stage_routes(call_, logical_ids_);
        ggml_backend_tensor_set(mapped_, mapped.data(), 0, mapped.size() * sizeof(int32_t));
        return GGML_STATUS_SUCCESS;
    }
    void reset() noexcept override {}
private:
    MoePreparedGraph & parent_;
    MoeEngine & engine_;
    CallId call_;
    ggml_tensor * ids_, * mapped_;
    ggml_backend_t backend_;
    ggml_backend_t source_backend_ = nullptr;
    bool evaluated_ = false;
    std::vector<int32_t> logical_ids_;
};

MoePreparedGraph::MoePreparedGraph(const core::ExecutionContext & execution, core::PreparedGraphConfig config,
    std::shared_ptr<core::GraphWorkspace> workspace)
    : graph_(execution, std::move(config)), workspace_(std::move(workspace)) {}
MoePreparedGraph::~MoePreparedGraph() { release(); }
core::ModuleBuildContext & MoePreparedGraph::build_context() {
    require_idle();
    if (allocated_) throw std::logic_error("cannot modify an allocated MoE graph");
    return graph_.build_context();
}
core::ModuleBuildContext & MoePreparedGraph::persistent_build_context() { require_idle(); return graph_.persistent_build_context(); }
MoePreparedGraph::GraphId MoePreparedGraph::add_graph(std::string label, size_t nodes) { require_idle(); return graph_.add_graph(std::move(label), nodes); }
MoePreparedGraph::GraphId MoePreparedGraph::add_entry_point(std::string label, GraphId allocation, size_t nodes) { require_idle(); return graph_.add_entry_point(std::move(label), allocation, nodes); }
ggml_cgraph * MoePreparedGraph::graph(GraphId id) const { return graph_.graph(id); }
size_t MoePreparedGraph::graph_count() const noexcept { return graph_.graph_count(); }
void MoePreparedGraph::allocate_persistent() { require_idle(); graph_.allocate_persistent(); }
void MoePreparedGraph::clear_persistent() { require_idle(); graph_.clear_persistent(); }
size_t MoePreparedGraph::persistent_bytes() const noexcept { return graph_.persistent_bytes(); }
size_t MoePreparedGraph::workspace_bytes() const noexcept { return graph_.workspace_bytes(); }
size_t MoePreparedGraph::primary_workspace_bytes() const noexcept {
    return scheduler_ ? ggml_backend_sched_get_buffer_size(scheduler_, execution().backend()) : graph_.workspace_bytes();
}
const core::ExecutionContext & MoePreparedGraph::execution() const noexcept { return graph_.execution(); }
void MoePreparedGraph::require_idle() const {
    if (active_) throw std::logic_error("cannot reprepare or overlap an active MoE graph sequence");
}
void MoePreparedGraph::declare_execution(GraphRequirements requirements) {
    require_idle(); if (allocated_) throw std::logic_error("MoE graph execution requirements are frozen");
    for (auto * backend : requirements.backends) require_backend(backend);
    for (const auto & placement : requirements.placements) place_nodes({placement.first}, placement.second);
    if (requirements.evaluation) {
        if (requirements_.evaluation && (requirements_.evaluation != requirements.evaluation || requirements_.evaluation_data != requirements.evaluation_data))
            throw std::invalid_argument("conflicting MoE parent graph evaluation callbacks");
        requirements_.evaluation = requirements.evaluation; requirements_.evaluation_data = requirements.evaluation_data;
    }
    requirements_.scheduler = requirements_.scheduler || requirements.scheduler;
}
void MoePreparedGraph::require_backend(ggml_backend_t backend) {
    require_idle();
    if (allocated_ || !backend) throw std::invalid_argument("backend declaration requires an unallocated MoE graph and a valid backend");
    if (std::find(requirements_.backends.begin(), requirements_.backends.end(), backend) == requirements_.backends.end()) requirements_.backends.push_back(backend);
}
void MoePreparedGraph::place_nodes(const std::vector<ggml_tensor *> & nodes, ggml_backend_t backend) {
    require_backend(backend);
    for (auto * node : nodes) {
        if (!node) throw std::invalid_argument("MoE placement requires a graph node");
        const auto found = std::find_if(requirements_.placements.begin(), requirements_.placements.end(), [&](const auto & item) { return item.first == node; });
        if (found != requirements_.placements.end()) {
            if (found->second != backend) throw std::invalid_argument("conflicting explicit graph placement");
        } else requirements_.placements.emplace_back(node, backend);
    }
}
void MoePreparedGraph::bind_call(MoeEngine & engine, CallId call) {
    require_idle();
    if (allocated_) throw std::logic_error("cannot add an expert call after graph allocation");
    if (std::none_of(calls_.begin(), calls_.end(), [&](const auto & item) { return item.first == &engine && item.second.index == call.index && item.second.generation == call.generation; }))
        calls_.emplace_back(&engine, call);
    if (std::find(engines_.begin(), engines_.end(), &engine) == engines_.end()) engines_.push_back(&engine);
}
MoePreparedGraph::SequenceId MoePreparedGraph::add_sequence(std::vector<Step> steps) {
    require_idle();
    if (allocated_ || steps.empty()) throw std::invalid_argument("MoE sequence requires steps before allocation");
    std::vector<BoundStep> bound;
    bound.reserve(steps.size());
    for (const auto & step : steps) {
        if (!step.expert) { (void) graph_.graph(step.graph); bound.push_back({step.graph, nullptr, {}}); continue; }
        const auto binding = std::find_if(calls_.begin(), calls_.end(), [&](const auto & item) {
            return item.second.owner == step.call.owner && item.second.index == step.call.index && item.second.generation == step.call.generation;
        });
        if (binding == calls_.end()) throw std::invalid_argument("sequence contains an unbound expert call");
        bound.push_back({0, binding->first, step.call});
    }
    sequences_.push_back(std::move(bound)); return {sequences_.size() - 1};
}
std::shared_ptr<core::GraphWorkspace> MoePreparedGraph::embedded_workspace() {
    require_idle();
    if (allocated_) throw std::logic_error("cannot add embedded expert graphs after allocation");
    if (!embedded_workspace_) embedded_workspace_ = std::make_shared<core::GraphWorkspace>(execution().backend());
    return embedded_workspace_;
}

void MoePreparedGraph::allocate() {
    require_idle(); if (allocated_) throw std::logic_error("MoE graph is already allocated");
    if (!cache_routes_.empty()) {
        if (!workspace_ && graph_count() == 1) {
            requirements_.scheduler = true;
            const bool host = core::is_host_backend(execution().backend()) ||
                std::any_of(requirements_.backends.begin(), requirements_.backends.end(), core::is_host_backend);
            if (!host) require_backend(cache_routes_.front()->engine_.routing_backend());
            for (const auto & route : cache_routes_) {
                route->evaluated_ = true;
                cache_evaluation_.emplace(route->mapped_, route.get());
            }
        } else record_cache_regions();
    }
    const bool other_backend = std::any_of(requirements_.backends.begin(), requirements_.backends.end(), [&](auto * backend) { return backend != execution().backend(); });
    if (other_backend || requirements_.scheduler) {
        if (workspace_ || graph_count() != 1) throw std::invalid_argument("scheduled MoE parent requires one graph and no shared workspace");
        std::vector<ggml_backend_t> backends{execution().backend()};
        for (auto * backend : requirements_.backends)
            if (backend != execution().backend() && !core::is_host_backend(backend)) backends.push_back(backend);
        for (auto * backend : requirements_.backends)
            if (backend != execution().backend() && core::is_host_backend(backend)) backends.push_back(backend);
        if (!core::is_host_backend(backends.back())) throw std::invalid_argument("scheduled MoE parent requires a declared CPU fallback backend");
        scheduler_ = graph_.create_scheduler(std::move(backends));
        for (const auto & placement : requirements_.placements) ggml_backend_sched_set_tensor_backend(scheduler_, placement.first, placement.second);
        if (!cache_evaluation_.empty()) ggml_backend_sched_set_eval_callback(scheduler_, evaluate, this);
        else if (requirements_.evaluation) ggml_backend_sched_set_eval_callback(scheduler_, requirements_.evaluation, requirements_.evaluation_data);
    } else if (requirements_.evaluation) throw std::invalid_argument("graph evaluation callback requires an explicit scheduler");
    if (embedded_workspace_ && std::any_of(calls_.begin(), calls_.end(), [](const auto & binding) {
        return binding.first->needs_primary_workspace(binding.second);
    })) embedded_workspace_->prepare();
    if (workspace_) { workspace_->add(graph_); workspace_->prepare(); }
    else graph_.allocate();
    for (const auto & binding : calls_) binding.first->prepare(binding.second);
    allocated_ = true;
}
void MoePreparedGraph::allocate_shared(const std::vector<MoePreparedGraph *> & owners) {
    if (owners.empty() || !owners.front()->workspace_) throw std::invalid_argument("shared MoE allocation requires graph owners and a workspace");
    auto workspace = owners.front()->workspace_;
    for (auto * owner : owners) {
        owner->require_idle();
        if (owner->requirements_.evaluation)
            throw std::invalid_argument("graph evaluation callback requires an explicit scheduler");
        if (owner->allocated_ || owner->workspace_ != workspace || owner->requirements_.scheduler ||
            std::any_of(owner->requirements_.backends.begin(), owner->requirements_.backends.end(),
                [&](auto * backend) { return backend != owner->execution().backend(); }))
            throw std::invalid_argument("incompatible MoE shared graph allocation");
    }
    for (auto * owner : owners) { owner->record_cache_regions(); workspace->add(owner->graph_); }
    workspace->prepare();
    for (auto * owner : owners) {
        for (const auto & binding : owner->calls_) binding.first->prepare(binding.second);
        owner->allocated_ = true;
    }
}
void MoePreparedGraph::compute(GraphId id) {
    if (calls_.empty()) { graph_.compute(id); return; }
    Scope scope(*this); graph_.compute(id); scope.finish();
}
void MoePreparedGraph::compute(SequenceId id) {
    const auto & steps = sequences_.at(id.index);
    Scope scope(*this);
    for (const auto & step : steps) {
        if (step.engine) step.engine->execute(step.call);
        else graph_.compute(step.graph);
    }
    scope.finish();
}
void MoePreparedGraph::bind_region(MoeEngine & engine, CallId call, ggml_tensor * input,
    ggml_tensor * ids, ggml_tensor * first, ggml_tensor * output) {
    bind_call(engine, call);
    core::GraphRegion region;
    region.outputs = {output};
    ggml_set_output(ids);
    const auto visit = [&](const auto & self, ggml_tensor * node) -> void {
        if (!node || node == input || node == ids || node->op == GGML_OP_NONE || std::find(region.nodes.begin(), region.nodes.end(), node) != region.nodes.end()) return;
        if (node == first->src[1]) return;
        for (auto * source : node->src) self(self, source);
        region.nodes.push_back(node);
    };
    visit(visit, output);
    graph_.record_region(std::move(region), [this, &engine, call, input, ids, output] {
        return std::make_unique<Region>(*this, engine, call, input, ids, output);
    });
}
void MoePreparedGraph::bind_cache_route(MoeEngine & engine, CallId call,
    ggml_tensor * ids, ggml_tensor * mapped, ggml_backend_t backend) {
    bind_call(engine, call);
    cache_routes_.push_back(std::make_unique<CacheRoute>(*this, engine, call, ids, mapped, backend));
}
void MoePreparedGraph::record_cache_regions() {
    for (const auto & route : cache_routes_) {
        ggml_set_output(route->ids_);
        core::GraphRegion region;
        region.nodes = {route->mapped_}; region.outputs = {route->mapped_};
        graph_.record_region(std::move(region), [this, route = route.get()] {
            return std::make_unique<CacheRoute>(*this, route->engine_, route->call_,
                route->ids_, route->mapped_, route->backend_);
        });
    }
}
bool MoePreparedGraph::evaluate(ggml_tensor * tensor, bool ask, void * data) {
    auto & owner = *static_cast<MoePreparedGraph *>(data);
    if (ask) {
        const auto found = owner.cache_evaluation_.find(tensor);
        owner.requested_route_ = found == owner.cache_evaluation_.end() ? nullptr : found->second;
        owner.caller_requested_ = owner.requirements_.evaluation &&
            owner.requirements_.evaluation(tensor, true, owner.requirements_.evaluation_data);
        return owner.requested_route_ || owner.caller_requested_;
    }
    if (owner.requested_route_) owner.requested_route_->compute();
    return !owner.caller_requested_ ||
        owner.requirements_.evaluation(tensor, false, owner.requirements_.evaluation_data);
}
void MoePreparedGraph::drain() {
    if (scheduler_) ggml_backend_sched_synchronize(scheduler_);
    else ggml_backend_synchronize(execution().backend());
}
void MoePreparedGraph::drop_calls() {
    for (const auto & binding : calls_) binding.first->release(binding.second);
    calls_.clear(); sequences_.clear(); engines_.clear(); embedded_workspace_.reset();
    cache_evaluation_.clear(); cache_routes_.clear(); requested_route_ = nullptr; caller_requested_ = false;
}
void MoePreparedGraph::reset() {
    require_idle(); drain();
    if (workspace_) workspace_->reset();
    graph_.reset(); drop_calls(); requirements_ = {}; scheduler_ = nullptr; allocated_ = false;
}
void MoePreparedGraph::release() {
    require_idle(); drain();
    if (workspace_) workspace_->reset();
    graph_.release(); drop_calls(); requirements_ = {}; scheduler_ = nullptr; allocated_ = false;
}
} // namespace engine::moe
