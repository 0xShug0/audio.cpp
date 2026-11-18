#pragma once

#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/moe_engine/moe_engine.h"

#include <atomic>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace engine::moe {

struct GraphRequirements {
    std::vector<ggml_backend_t> backends;
    std::vector<std::pair<ggml_tensor *, ggml_backend_t>> placements;
    ggml_backend_sched_eval_callback evaluation = nullptr;
    void * evaluation_data = nullptr;
    bool scheduler = false;
};

// Complete parent-graph preparation and execution owner. Expert bindings borrow
// engine calls; engines never retain this owner or callbacks to its completion.
class MoePreparedGraph {
public:
    using GraphId = core::PreparedGraph::GraphId;
    struct SequenceId { size_t index; };
    struct Step {
        GraphId graph = 0;
        CallId call{};
        bool expert = false;
        Step(GraphId id) : graph(id) {}
        Step(CallId id) : call(id), expert(true) {}
    };
    MoePreparedGraph(const core::ExecutionContext &, core::PreparedGraphConfig,
        std::shared_ptr<core::GraphWorkspace> workspace = {});
    ~MoePreparedGraph();
    MoePreparedGraph(const MoePreparedGraph &) = delete;
    MoePreparedGraph & operator=(const MoePreparedGraph &) = delete;

    core::ModuleBuildContext & build_context();
    core::ModuleBuildContext & persistent_build_context();
    GraphId add_graph(std::string label, size_t nodes = 0);
    GraphId add_entry_point(std::string label, GraphId allocation, size_t nodes = 0);
    ggml_cgraph * graph(GraphId id = 0) const;
    size_t graph_count() const noexcept;
    void allocate_persistent();
    void clear_persistent();
    size_t workspace_bytes() const noexcept;
    size_t persistent_bytes() const noexcept;
    size_t primary_workspace_bytes() const noexcept;
    const core::ExecutionContext & execution() const noexcept;

    void declare_execution(GraphRequirements requirements);
    void require_backend(ggml_backend_t backend);
    void place_nodes(const std::vector<ggml_tensor *> &, ggml_backend_t);
    SequenceId add_sequence(std::vector<Step> steps);
    void allocate();
    static void allocate_shared(const std::vector<MoePreparedGraph *> & owners);
    void compute(GraphId id = 0);
    void compute(SequenceId sequence);
    void reset();
    void release();

private:
    friend class MoeEngine;
    void bind_call(MoeEngine &, CallId);
    class Region;
    class CacheRoute;
    class Scope;
    std::shared_ptr<core::GraphWorkspace> embedded_workspace();
    void bind_region(MoeEngine &, CallId, ggml_tensor * input,
        ggml_tensor * ids, ggml_tensor * first, ggml_tensor * output);
    void bind_cache_route(MoeEngine &, CallId, ggml_tensor * ids,
        ggml_tensor * mapped, ggml_backend_t backend);
    void record_cache_regions();
    static bool evaluate(ggml_tensor *, bool ask, void * data);
    void drain();
    void drop_calls();
    void require_idle() const;
    core::PreparedGraph graph_;
    std::shared_ptr<core::GraphWorkspace> workspace_, embedded_workspace_;
    GraphRequirements requirements_;
    ggml_backend_sched_t scheduler_ = nullptr;
    std::vector<std::pair<MoeEngine *, CallId>> calls_;
    struct BoundStep { GraphId graph; MoeEngine * engine; CallId call; };
    std::vector<std::vector<BoundStep>> sequences_;
    std::vector<MoeEngine *> engines_;
    std::vector<std::unique_ptr<CacheRoute>> cache_routes_;
    std::unordered_map<ggml_tensor *, CacheRoute *> cache_evaluation_;
    CacheRoute * requested_route_ = nullptr;
    bool caller_requested_ = false;
    std::atomic<bool> active_{false};
    bool allocated_ = false;
};
} // namespace engine::moe
