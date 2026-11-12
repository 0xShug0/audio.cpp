#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/core/graph_workspace.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <unordered_set>

namespace engine::core {

struct PreparedGraph::GraphEntry {
    struct Region {
        GraphRegion definition;
        std::unique_ptr<GraphRegionExecutor> executor;
        bool active = false;
        static ggml_status compute(ggml_backend_t, ggml_cgraph *, void * data) {
            return static_cast<Region *>(data)->executor->compute();
        }
    };
    struct Segment {
        ggml_cgraph * graph;
        Region * region;
        HostGraphPlan plan;
    };
    std::string label;
    ggml_cgraph * graph = nullptr;
    GraphId allocation_graph = 0;
    HostGraphPlan plan;
    std::vector<std::unique_ptr<Region>> regions;
    std::vector<std::unique_ptr<Segment>> segments;
};

PreparedGraph::PreparedGraph(const ExecutionContext & execution, PreparedGraphConfig config)
    : execution_(execution), config_(std::move(config)) {
    if (config_.label.empty() || config_.metadata_bytes == 0 || config_.graph_nodes == 0)
        throw std::runtime_error("PreparedGraph requires a label and positive capacities");
    reset();
}

PreparedGraph::~PreparedGraph() {
    if (workspace_) workspace_->forget(*this);
    release();
}

ggml_cgraph * PreparedGraph::graph(GraphId id) const {
    return graphs_.at(id)->graph;
}

bool PreparedGraph::has_host_plan(GraphId id) const {
    return graphs_.at(id)->plan.active();
}

PreparedGraph::GraphId PreparedGraph::add_graph(std::string name, size_t graph_nodes) {
    if (!context_ || workspace_ || scheduler_ || allocation_started_)
        throw std::logic_error(config_.label + " graphs can only be added before preparation");
    if (name.empty()) throw std::invalid_argument("Prepared graph name must not be empty");
    const std::string label = config_.label + "." + name;
    for (const auto & entry : graphs_) {
        if (entry->label == label) throw std::invalid_argument("Duplicate prepared graph name: " + name);
    }
    auto entry = std::make_unique<GraphEntry>();
    entry->label = label;
    entry->allocation_graph = graphs_.size();
    entry->graph = ggml_new_graph_custom(context_.get(), graph_nodes ? graph_nodes : config_.graph_nodes, false);
    graphs_.push_back(std::move(entry));
    return graphs_.size() - 1;
}

PreparedGraph::GraphId PreparedGraph::add_entry_point(std::string name, GraphId allocation_graph, size_t graph_nodes) {
    if (graphs_.at(allocation_graph)->allocation_graph != allocation_graph)
        throw std::invalid_argument("Entry points must reference an allocation graph, not another entry point");
    const auto id = add_graph(std::move(name), graph_nodes);
    graphs_[id]->allocation_graph = allocation_graph;
    return id;
}

PreparedGraph::GraphId PreparedGraph::allocation_graph(GraphId id) const {
    return graphs_.at(id)->allocation_graph;
}

ModuleBuildContext & PreparedGraph::persistent_build_context() {
    if (persistent_buffer_ || workspace_ || allocation_started_)
        throw std::logic_error(config_.label + " persistent tensors must be declared before allocation");
    if (config_.persistent_metadata_bytes == 0)
        throw std::logic_error(config_.label + " requires persistent metadata capacity");
    if (!persistent_context_) {
        persistent_context_.reset(ggml_init({config_.persistent_metadata_bytes, nullptr, true}));
        if (!persistent_context_) throw std::runtime_error("Cannot allocate " + config_.label + " persistent metadata");
        persistent_build_context_ = {persistent_context_.get(), config_.label.c_str(), execution_.backend_type()};
    }
    return persistent_build_context_;
}

void PreparedGraph::allocate_persistent() {
    if (!persistent_context_ || !ggml_get_first_tensor(persistent_context_.get()) ||
        persistent_buffer_ || workspace_ || allocation_started_)
        throw std::logic_error(config_.label + " persistent allocation requires declared, unallocated tensors");
    persistent_buffer_.reset(ggml_backend_alloc_ctx_tensors(persistent_context_.get(), execution_.backend()));
    if (!persistent_buffer_) throw std::runtime_error("Cannot allocate " + config_.label + " persistent tensors");
    debug::timing_log_scalar(config_.label + ".persistent_buffer_mb", persistent_bytes() / (1024.0 * 1024.0));
}

size_t PreparedGraph::persistent_bytes() const noexcept {
    return persistent_buffer_ ? ggml_backend_buffer_get_size(persistent_buffer_.get()) : 0;
}

size_t PreparedGraph::workspace_bytes() const noexcept {
    if (scheduler_) {
        size_t bytes = 0;
        for (auto * backend : scheduler_backends_) bytes += ggml_backend_sched_get_buffer_size(scheduler_.get(), backend);
        return bytes;
    }
    return workspace_ ? workspace_->bytes() : (owned_workspace_ ? owned_workspace_->bytes() : 0);
}

void PreparedGraph::clear_persistent() {
    if (!persistent_buffer_) throw std::logic_error(config_.label + " persistent tensors are not allocated");
    synchronize();
    ggml_backend_buffer_clear(persistent_buffer_.get(), 0);
}

ggml_backend_sched_t PreparedGraph::create_scheduler(std::vector<ggml_backend_t> backends) {
    if (!context_ || workspace_ || scheduler_ || allocation_started_ || graphs_.size() != 1)
        throw std::logic_error(config_.label + " scheduler requires one unallocated graph and no workspace");
    if (backends.empty() || backends.front() != execution_.backend())
        throw std::invalid_argument("Scheduler backends must start with the prepared graph's primary backend");
    for (auto * backend : backends) {
        if (!backend) throw std::invalid_argument("Scheduler backend must not be null");
    }
    if (ggml_backend_dev_type(ggml_backend_get_device(backends.back())) != GGML_BACKEND_DEVICE_TYPE_CPU)
        throw std::invalid_argument("Scheduler requires a CPU backend last");
    // Changing allocator modes must not retain an unused second scratch arena.
    owned_workspace_.reset();
    cleanup_context_.reset(ggml_init({ggml_graph_overhead_custom(1, false) + 1024, nullptr, true}));
    if (!cleanup_context_) throw std::runtime_error("Cannot allocate " + config_.label + " scheduler cleanup metadata");
    cleanup_graph_ = ggml_new_graph_custom(cleanup_context_.get(), 1, false);
    scheduler_backends_ = std::move(backends);
    scheduler_.reset(ggml_backend_sched_new(scheduler_backends_.data(), nullptr,
        static_cast<int>(scheduler_backends_.size()), config_.graph_nodes, false, false));
    if (!scheduler_) throw std::runtime_error("Cannot create " + config_.label + " scheduler");
    return scheduler_.get();
}

void PreparedGraph::replace_region(GraphId id, GraphRegion definition,
    std::unique_ptr<GraphRegionExecutor> executor) {
    if (!context_ || workspace_ || allocation_started_)
        throw std::logic_error(config_.label + " regions must be registered before preparation");
    if (definition.nodes.empty() || definition.outputs.empty() || !executor)
        throw std::invalid_argument("Graph region requires exact nodes, outputs and an executor");
    if (!graphs_.at(id)->segments.empty()) throw std::logic_error("Reset the graph before changing prepared regions");
    graphs_.at(id)->regions.push_back(std::make_unique<GraphEntry::Region>(
        GraphEntry::Region{std::move(definition), std::move(executor), false}));
}

void PreparedGraph::record_region(GraphRegion definition,
    std::function<std::unique_ptr<GraphRegionExecutor>()> factory) {
    if (!context_ || workspace_ || allocation_started_ || regions_bound_)
        throw std::logic_error(config_.label + " regions must be recorded during graph construction");
    if (definition.nodes.empty() || definition.outputs.empty() || !factory)
        throw std::invalid_argument("Recorded region requires exact nodes, outputs and a factory");
    recorded_regions_.push_back({std::move(definition), std::move(factory)});
}

void PreparedGraph::bind_regions() {
    if (regions_bound_ || recorded_regions_.empty()) return;
    for (const auto & recorded : recorded_regions_) {
        const auto & definition = recorded.definition;
        const std::unordered_set<ggml_tensor *> members(definition.nodes.begin(), definition.nodes.end());
        bool bound = false;
        for (size_t id = 0; id < graphs_.size(); ++id) {
            size_t found = 0;
            for (int n = 0; n < ggml_graph_n_nodes(graph(id)); ++n)
                found += members.count(ggml_graph_node(graph(id), n));
            if (!found) continue;
            if (found != definition.nodes.size())
                throw std::invalid_argument(graphs_[id]->label + " contains only part of a recorded region");
            replace_region(id, definition, recorded.factory());
            bound = true;
        }
        if (!bound) throw std::invalid_argument(config_.label + " recorded region is absent");
    }
    regions_bound_ = true;
}

void PreparedGraph::build_segments(GraphEntry & entry) {
    if (!entry.segments.empty()) return;
    struct Range { int begin, end; GraphEntry::Region * region; };
    std::vector<Range> ranges;
    std::unordered_map<ggml_tensor *, GraphEntry::Region *> owners;
    std::unordered_set<ggml_tensor *> outputs;
    for (auto & region : entry.regions) {
        const auto expected = region->definition.nodes;
        int begin = -1;
        for (int n = 0; n < ggml_graph_n_nodes(entry.graph); ++n)
            if (ggml_graph_node(entry.graph, n) == expected.front()) { begin = n; break; }
        if (begin < 0) throw std::invalid_argument(entry.label + " region nodes are absent");
        std::unordered_set<ggml_tensor *> operands;
        for (auto * node : expected) {
            if (!node) throw std::invalid_argument(entry.label + " null region node");
            for (auto * source : node->src) if (source) operands.insert(source);
        }
        std::vector<ggml_tensor *> nodes;
        int end = begin;
        for (auto * node : expected) {
            while (end < ggml_graph_n_nodes(entry.graph) && ggml_graph_node(entry.graph, end) != node) {
                auto * marker = ggml_graph_node(entry.graph, end);
                // GGML places PARAM leaves in its node list on their first use.
                // These already allocated operands perform no work; retain their
                // markers, but never absorb an unrelated computational node.
                if (marker->op != GGML_OP_NONE || !operands.count(marker))
                    throw std::invalid_argument(entry.label + " region contains an undeclared operation " + marker->name);
                nodes.push_back(marker);
                ++end;
            }
            if (end == ggml_graph_n_nodes(entry.graph))
                throw std::invalid_argument(entry.label + " region nodes are absent or out of order");
            nodes.push_back(node);
            ++end;
        }
        for (auto * node : nodes)
            if (!owners.emplace(node, region.get()).second)
                throw std::invalid_argument(entry.label + " graph regions overlap");
        region->definition.nodes = std::move(nodes);
        const auto & sealed = region->definition.nodes;
        for (auto * output : region->definition.outputs) {
            if (!output || std::find(sealed.begin(), sealed.end(), output) == sealed.end())
                throw std::invalid_argument(entry.label + " region output is not a member");
            outputs.insert(output);
        }
        ranges.push_back({begin, end, region.get()});
    }
    // An executor must not silently discard an intermediate used by another
    // graph operation, a graph output or an entry point's external consumer.
    for (int n = 0; n < ggml_graph_n_nodes(entry.graph); ++n) {
        auto * node = ggml_graph_node(entry.graph, n);
        const auto owner = owners.find(node);
        if (node->op != GGML_OP_NONE && owner != owners.end() && (node->flags & GGML_TENSOR_FLAG_OUTPUT) && !outputs.count(node))
            throw std::invalid_argument(entry.label + " region has an undeclared graph output");
        for (auto * source : node->src) {
            const auto producer = owners.find(source);
            if (producer != owners.end() && source->op != GGML_OP_NONE && (owner == owners.end() || owner->second != producer->second) &&
                !outputs.count(source))
                throw std::invalid_argument(entry.label + " region has an undeclared escaping value");
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range & a, const Range & b) { return a.begin < b.begin; });
    int begin = 0;
    entry.segments.reserve(2 * ranges.size() + 1);
    for (const auto & range : ranges) {
        if (range.begin > begin)
            entry.segments.push_back(std::make_unique<GraphEntry::Segment>(GraphEntry::Segment{
                ggml_graph_slice(context_.get(), entry.graph, begin, range.begin), nullptr, {}}));
        entry.segments.push_back(std::make_unique<GraphEntry::Segment>(GraphEntry::Segment{
            ggml_graph_slice(context_.get(), entry.graph, range.begin, range.end), range.region, {}}));
        begin = range.end;
    }
    if (begin < ggml_graph_n_nodes(entry.graph))
        entry.segments.push_back(std::make_unique<GraphEntry::Segment>(GraphEntry::Segment{
            ggml_graph_slice(context_.get(), entry.graph, begin, ggml_graph_n_nodes(entry.graph)), nullptr, {}}));
}

void PreparedGraph::prepare_segments(GraphEntry & entry) {
    for (auto & segment : entry.segments) {
        if (segment->region) {
            auto * backend = scheduler_ ? ggml_backend_sched_get_tensor_backend(scheduler_.get(), segment->region->definition.nodes.front())
                : execution_.backend();
            segment->region->active = true; // reset also cleans up a partially failed prepare().
            segment->region->executor->prepare(backend, segment->graph);
        } else if (!scheduler_) prepare_host_graph_plan(execution_, segment->graph, segment->plan);
    }
}

void PreparedGraph::release_segments(GraphEntry & entry) {
    for (auto & region : entry.regions) {
        if (region->active) region->executor->reset();
        region->active = false;
    }
    for (auto & segment : entry.segments) {
        segment->plan.reset();
        if (!scheduler_) release_backend_graph_resources(execution_.backend(), segment->graph, true);
    }
}

void PreparedGraph::synchronize() {
    if (scheduler_) ggml_backend_sched_synchronize(scheduler_.get());
    else ggml_backend_synchronize(execution_.backend());
}

void PreparedGraph::release_graphs() {
    for (auto & entry : graphs_) release_segments(*entry);
    // Scheduler split views borrow nodes from the allocation graph. Visit every
    // possible first-node key while metadata and split buffers are still alive.
    if (scheduler_) {
        for (int node = 0; node < ggml_graph_n_nodes(graph()); ++node) {
            ggml_graph_set_n_nodes(cleanup_graph_, 0);
            ggml_graph_add_node(cleanup_graph_, ggml_graph_node(graph(), node));
            for (auto * backend : scheduler_backends_)
                release_backend_graph_resources(backend, cleanup_graph_, true);
        }
        scheduler_.reset();
    }
    cleanup_graph_ = nullptr;
    cleanup_context_.reset();
    scheduler_backends_.clear();
    for (auto & entry : graphs_) {
        entry->plan.reset();
        if (entry->graph) release_backend_graph_resources(execution_.backend(), entry->graph, true);
        entry->graph = nullptr;
    }
    if (owned_workspace_) owned_workspace_->reset();
    scratch_tensors_.clear();
    releasable_ = false;
    allocation_started_ = false;
    allocated_ = false;
}

void PreparedGraph::reset() {
    if (workspace_) throw std::logic_error(config_.label + " is still registered with a workspace; reset the workspace first");
    if (!graphs_.empty()) synchronize();
    release_graphs();
    build_context_ = {};
    graphs_.clear();
    recorded_regions_.clear();
    region_states_.reset();
    regions_bound_ = false;
    if (context_) ggml_reset(context_.get());
    else context_.reset(ggml_init({config_.metadata_bytes, nullptr, true}));
    if (!context_) throw std::runtime_error("Cannot allocate " + config_.label + " graph metadata");
    build_context_ = {context_.get(), config_.label.c_str(), execution_.backend_type()};
    auto entry = std::make_unique<GraphEntry>();
    entry->label = config_.label;
    entry->graph = ggml_new_graph_custom(context_.get(), config_.graph_nodes, false);
    graphs_.push_back(std::move(entry));
    if (execution_.moe_cpu_share() > 0 && !is_host_backend(execution_.backend())) {
        build_context_.region_owner = this;
    }
}

void PreparedGraph::release() {
    if (workspace_) throw std::logic_error(config_.label + " is still registered with a workspace; reset the workspace first");
    if (!graphs_.empty() || persistent_buffer_) synchronize();
    release_graphs();
    decltype(graphs_){}.swap(graphs_);
    recorded_regions_.clear();
    region_states_.reset();
    regions_bound_ = false;
    owned_workspace_.reset();
    context_.reset();
    build_context_ = {};
    persistent_buffer_.reset();
    persistent_context_.reset();
    persistent_build_context_ = {};
}

void PreparedGraph::allocate_releasable() {
    if (!context_ || workspace_ || scheduler_ || allocation_started_)
        throw std::logic_error(config_.label + " releasable allocation requires an unallocated, owned graph set");
    scratch_tensors_.clear();
    for (auto * tensor = ggml_get_first_tensor(context_.get()); tensor; tensor = ggml_get_next_tensor(context_.get(), tensor)) {
        if (!tensor->view_src && !tensor->data) scratch_tensors_.push_back(tensor);
    }
    allocate();
    releasable_ = true;
}

void PreparedGraph::release_buffers() {
    if (workspace_ || scheduler_ || !releasable_ || !allocated_)
        throw std::logic_error(config_.label + " buffer release requires an allocated releasable graph set");
    synchronize();
    for (auto & entry : graphs_) {
        release_segments(*entry);
        entry->plan.reset();
        release_backend_graph_resources(execution_.backend(), entry->graph, true);
    }
    std::unordered_set<ggml_backend_buffer_t> buffers;
    if (persistent_buffer_) buffers.insert(persistent_buffer_.get());
    for (const auto * tensor : scratch_tensors_) if (tensor->buffer) buffers.insert(tensor->buffer);
    for (auto * context : {context_.get(), persistent_context_.get()}) {
        if (!context) continue;
        for (auto * tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor)) {
            if (buffers.count(tensor->buffer)) {
                tensor->data = nullptr;
                tensor->buffer = nullptr;
                tensor->extra = nullptr;
            }
        }
    }
    owned_workspace_.reset();
    persistent_buffer_.reset();
    scratch_tensors_.clear();
    releasable_ = false;
    allocation_started_ = false;
    allocated_ = false;
}

void PreparedGraph::allocate() {
    if (graphs_.empty() || !graph() || workspace_ || allocation_started_)
        throw std::runtime_error(config_.label + " allocation requires unallocated, unregistered graphs; reset before rebuilding");
    if (persistent_context_ && !persistent_buffer_)
        throw std::logic_error(config_.label + " persistent tensors must be allocated before graph preparation");
    bind_regions();
    if (scheduler_) {
        if (!graphs_.front()->regions.empty()) {
            build_segments(*graphs_.front());
            std::vector<ggml_backend_graph_region> regions;
            for (const auto & region : graphs_.front()->regions)
                regions.push_back({region->definition.nodes.data(), region->definition.nodes.size(),
                    region->definition.outputs.data(), region->definition.outputs.size(),
                    region->definition.deferred_eval.data(), region->definition.deferred_eval.size(),
                    GraphEntry::Region::compute, region.get()});
            if (!ggml_backend_sched_set_regions(scheduler_.get(), regions.data(), regions.size()))
                throw std::runtime_error(config_.label + " cannot register scheduled regions");
        }
        // Validate explicit placement hints. Unassigned nodes are placed by GGML,
        // which can choose another backend based on support and weight residency.
        for (int node = 0; node < ggml_graph_n_nodes(graph()); ++node) {
            auto * tensor = ggml_graph_node(graph(), node);
            auto * backend = ggml_backend_sched_get_tensor_backend(scheduler_.get(), tensor);
            if (backend && !ggml_backend_supports_op(backend, tensor))
                throw std::runtime_error(config_.label + " scheduled backend does not support " + ggml_op_name(tensor->op));
        }
        allocation_started_ = true;
        if (!ggml_backend_sched_alloc_graph(scheduler_.get(), graph()))
            throw std::runtime_error("Cannot allocate " + config_.label + " scheduled graph");
        allocated_ = true;
        prepare_segments(*graphs_.front());
    } else {
        validate_graphs();
        allocation_started_ = true;
        if (!owned_workspace_) owned_workspace_ = std::make_unique<GraphWorkspace>(execution_.backend());
        for (size_t id = 0; id < graphs_.size(); ++id) {
            if (allocation_graph(id) == id) owned_workspace_->add(graph(id));
        }
        owned_workspace_->prepare();
        prepare_shared_allocation();
    }
    debug::timing_log_scalar(config_.label + ".buffer_mb", workspace_bytes() / (1024.0 * 1024.0));
}

void PreparedGraph::validate_graphs() const {
    // Independent allocation graphs receive independent scratch layouts. Entry
    // points share the allocation graph's existing layout and are never allocated
    // separately. Validate all of this before touching allocation state.
    std::unordered_map<const ggml_tensor *, size_t> scratch_owners;
    for (size_t id = 0; id < graphs_.size(); ++id) {
        const auto & entry = *graphs_[id];
        validate_backend_graph_supported(execution_.backend(), entry.graph, entry.label.c_str());
        if (entry.allocation_graph != id) {
            auto * allocation = graph(entry.allocation_graph);
            int position = 0;
            for (int node = 0; node < ggml_graph_n_nodes(entry.graph); ++node) {
                while (position < ggml_graph_n_nodes(allocation) &&
                    ggml_graph_node(allocation, position) != ggml_graph_node(entry.graph, node)) ++position;
                if (position == ggml_graph_n_nodes(allocation))
                    throw std::logic_error(entry.label + " entry point must be an ordered subset of its allocation graph");
                ++position;
            }
        }
        std::unordered_set<const ggml_tensor *> visited;
        std::vector<const ggml_tensor *> pending;
        for (int node = 0; node < ggml_graph_n_nodes(entry.graph); ++node)
            pending.push_back(ggml_graph_node(entry.graph, node));
        while (!pending.empty()) {
            const auto * tensor = pending.back();
            pending.pop_back();
            if (!visited.insert(tensor).second) continue;
            const auto * storage = tensor;
            while (storage->view_src) storage = storage->view_src;
            if (!storage->data) {
                const auto inserted = scratch_owners.emplace(tensor, entry.allocation_graph);
                if (!inserted.second && inserted.first->second != entry.allocation_graph)
                    throw std::logic_error(config_.label + " graphs share scratch tensor '" + tensor->name +
                        "'; use persistent storage for cross-graph values");
            }
            for (const auto * source : tensor->src) {
                if (source) pending.push_back(source);
            }
        }
    }
}

void PreparedGraph::prepare_shared_allocation() {
    for (auto & entry : graphs_) {
        if (entry->regions.empty()) prepare_host_graph_plan(execution_, entry->graph, entry->plan);
        else {
            build_segments(*entry);
            prepare_segments(*entry);
        }
    }
    allocation_started_ = true;
    allocated_ = true;
}

void PreparedGraph::compute(GraphId id) {
    if (!allocated_) throw std::runtime_error(config_.label + " graphs are not allocated");
    auto & entry = *graphs_.at(id);
    if (!scheduler_ && !entry.regions.empty()) {
        for (auto & segment : entry.segments) {
            const auto status = segment->region ? segment->region->executor->compute()
                : compute_graph(execution_, segment->graph, segment->plan, entry.label.c_str());
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error(entry.label + " graph region execution failed");
        }
        return;
    }
    const auto status = scheduler_ ? ggml_backend_sched_graph_compute(scheduler_.get(), entry.graph)
        : compute_graph(execution_, entry.graph, entry.plan, entry.label.c_str());
    if (status != GGML_STATUS_SUCCESS)
        throw std::runtime_error(entry.label + " graph execution failed");
}

}  // namespace engine::core
