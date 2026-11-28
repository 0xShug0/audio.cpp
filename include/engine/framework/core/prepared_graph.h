#pragma once

#include "engine/framework/core/execution_context.h"

#include <ggml-alloc.h>

#include <memory>
#include <functional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::core {

class GraphWorkspace;

// Exact, ordered operations replaced by an executor. Outputs are the only
// region values allowed to escape. Preparation retains non-executing PARAM
// operand markers interleaved by GGML. Deferred evaluation callbacks are inputs
// whose handling belongs to the executor rather than the original callback.
struct GraphRegion {
    std::vector<ggml_tensor *> nodes;
    std::vector<ggml_tensor *> outputs;
    std::vector<ggml_tensor *> deferred_eval;
};

// An explicitly substituted execution region, owned by its PreparedGraph.
// prepare() sees the original allocated nodes and their assigned backend.
// compute() must produce all externally consumed values and preserve the
// original aliasing/side effects. It may enqueue work on the supplied backend;
// host consumers must drain dependencies themselves. reset() drains/releases
// executor-owned work and resources before graph storage is freed or relocated.
class GraphRegionExecutor {
public:
    virtual ~GraphRegionExecutor() = default;
    virtual void prepare(ggml_backend_t backend, ggml_cgraph * region) = 0;
    virtual ggml_status compute() = 0;
    virtual void reset() noexcept = 0;
};

struct PreparedGraphConfig {
    std::string label;
    size_t metadata_bytes = 0;
    size_t graph_nodes = 0;
    size_t persistent_metadata_bytes = 0;
};

// Session-owned graph set. All graphs share metadata and execute serially using
// one scratch workspace. Persistent tensors have separate metadata/storage and
// survive reset(). ExecutionContext must outlive this owner. Graph IDs, tensor
// handles and build contexts for transient metadata are invalid after reset().
// Preallocated external tensors can also be referenced; their context/buffer
// must outlive every graph using them. This owner never frees external storage.
class PreparedGraph {
public:
    using GraphId = size_t;

    PreparedGraph(const ExecutionContext & execution, PreparedGraphConfig config);
    ~PreparedGraph();
    PreparedGraph(const PreparedGraph &) = delete;
    PreparedGraph & operator=(const PreparedGraph &) = delete;

    ModuleBuildContext & build_context() noexcept { return build_context_; }
    // Graph zero exists at construction/reset; additional graphs share its context.
    GraphId add_graph(std::string name, size_t graph_nodes = 0);
    // An execution-only entry point shares the allocation graph's tensors and
    // layout. Its nodes must be an ordered subset of that allocation graph.
    // Build both graphs before allocation; only the allocation graph is allocated.
    GraphId add_entry_point(std::string name, GraphId allocation_graph, size_t graph_nodes = 0);
    GraphId allocation_graph(GraphId id = 0) const;
    ggml_cgraph * graph(GraphId id = 0) const;
    size_t graph_count() const noexcept { return graphs_.size(); }
    const ExecutionContext & execution() const noexcept { return execution_; }
    const std::string & label() const noexcept { return config_.label; }
    bool allocated() const noexcept { return allocated_; }
    bool has_host_plan(GraphId id = 0) const;

    // Declare persistent inputs/state before allocation. Initialize their values
    // with the existing tensor-write APIs; allocation never initializes state.
    // Use these tensors across independent allocation graphs. Entry points of
    // the same allocation graph may share its transient tensors directly.
    ModuleBuildContext & persistent_build_context();
    void allocate_persistent();
    size_t persistent_bytes() const noexcept;
    size_t workspace_bytes() const noexcept;
    // Explicit zero fill, not a model-specific state initializer. Drains pending
    // work first; graph rebuild/reset never changes persistent tensor contents.
    void clear_persistent();

    // Finish graph construction before calling once. Prepares every graph and
    // reserves shared scratch. Structure/addresses stay fixed until reset().
    // Persistent tensors must already be allocated before preparing the graphs.
    void allocate();
    // Opt-in allocation that can later discard device buffers while retaining
    // graph metadata. External tensor storage is never owned or released here.
    void allocate_releasable();
    // Drains execution and releases plans, captures, scratch and persistent
    // buffers. Graph IDs and tensor headers survive, but tensor data is invalid.
    // Reallocate persistent storage, initialize it, then allocate_releasable()
    // before executing again. Shared workspaces and schedulers are unsupported.
    void release_buffers();
    // Optional single-graph scheduler execution. The returned handle is borrowed:
    // configure placement/evaluation with GGML's APIs before allocate(). Backend
    // handles (primary first, CPU last) and callback data must outlive this owner. A scheduler
    // owns its split allocations, so it cannot share a GraphWorkspace or entry points.
    ggml_backend_sched_t create_scheduler(std::vector<ggml_backend_t> backends);
    // Replace exact, contiguous operations before preparation. The full
    // original graph still determines scratch layout/liveness. Entry points
    // register their own replacements. Scheduled regions must fit one split.
    // Without registration, the original preparation/execution path is used.
    void replace_region(GraphId id, GraphRegion region,
        std::unique_ptr<GraphRegionExecutor> executor);
    // Modules record boundaries while building tensors, before entry points
    // have been assembled. Factories bind to complete execution entry points.
    void record_region(GraphRegion region,
        std::function<std::unique_ptr<GraphRegionExecutor>()> factory);
    // Share compilation/execution state between a module's recorded regions.
    // Created on demand, retained across buffer release, destroyed on reset.
    template<class State, class... Args> std::shared_ptr<State> region_state(Args &&... args) {
        if (!region_states_) region_states_ = std::make_unique<RegionStates>();
        auto & state = (*region_states_)[std::type_index(typeid(State))];
        if (!state) state = std::make_shared<State>(std::forward<Args>(args)...);
        return std::static_pointer_cast<State>(state);
    }
    void compute(GraphId id = 0);
    // Drain work, release plans/captures and rebuild empty transient metadata.
    // Retains persistent state and owned scratch capacity. For an external
    // workspace, reset that workspace before resetting this owner.
    void reset();
    // Drain work and free all resources, including persistent state and scratch.
    // Idempotent; call reset() to start building again after release().
    void release();

private:
    friend class GraphWorkspace;
    struct GraphEntry;
    struct RecordedRegion {
        GraphRegion definition;
        std::function<std::unique_ptr<GraphRegionExecutor>()> factory;
    };
    using RegionStates = std::unordered_map<std::type_index, std::shared_ptr<void>>;
    void bind_regions();
    void validate_graphs() const;
    void prepare_shared_allocation();
    void build_segments(GraphEntry & entry);
    void prepare_segments(GraphEntry & entry);
    void release_segments(GraphEntry & entry);
    void release_graphs();
    void synchronize();

    const ExecutionContext & execution_;
    PreparedGraphConfig config_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> persistent_context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> persistent_buffer_{nullptr, ggml_backend_buffer_free};
    ModuleBuildContext build_context_;
    ModuleBuildContext persistent_build_context_;
    std::vector<std::unique_ptr<GraphEntry>> graphs_;
    std::vector<RecordedRegion> recorded_regions_;
    std::unique_ptr<RegionStates> region_states_;
    bool regions_bound_ = false;
    std::unique_ptr<GraphWorkspace> owned_workspace_;
    GraphWorkspace * workspace_ = nullptr;
    std::vector<ggml_backend_t> scheduler_backends_;
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> scheduler_{nullptr, ggml_backend_sched_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> cleanup_context_{nullptr, ggml_free};
    ggml_cgraph * cleanup_graph_ = nullptr;
    bool allocation_started_ = false;
    bool allocated_ = false;
    bool releasable_ = false;
    std::vector<ggml_tensor *> scratch_tensors_;
};

}  // namespace engine::core
