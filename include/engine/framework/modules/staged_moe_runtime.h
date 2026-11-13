#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/graph_workspace.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/modules/moe_module.h"

#include <memory>
#include <vector>

namespace engine::modules {

// Opt-in execution policy, separate from weight residency. GPU hits precede
// misses, so bounded staging cannot evict a hit that has not been consumed.
// Storage is reserved once; updates reuse it. Cache invalidation resets credit.
class MoePlacementPlan {
public:
    explicit MoePlacementPlan(size_t experts);
    void update(const std::vector<int32_t> & ids, const std::vector<int32_t> & resident,
        uint64_t cache_epoch, float cpu_share);
    std::vector<int32_t> gpu;
    std::vector<int32_t> cpu;
private:
    std::vector<int32_t> misses_;
    std::vector<uint8_t> seen_;
    uint64_t epoch_ = 0;
    double cpu_carry_ = 0;
};

struct StagedMoeConfig {
    int64_t hidden_size = 0;
    int64_t tokens = 0;
    int64_t selected = 0;
    int64_t experts = 0;
    int64_t cache_slots = 0;
    // Execution buckets, independent of model identity. Preserve caller lowering.
    std::vector<int64_t> rows_per_expert;
    ggml_type scatter_index_type = GGML_TYPE_I64;
    // Zero uses hidden_size. Separate width supports padded expert inputs.
    int64_t input_size = 0;
};

// Session-owned staged expert execution. Routing and surrounding backbone graphs
// are owned by the caller. Expert graph resources live here; activation scratch
// can be shared with serial backbone phases.
class StagedMoeRuntime {
public:
    struct Statistics {
        uint64_t cpu_assignments = 0;
        uint64_t gpu_assignments = 0;
        double cpu_ms = 0;
    };
    // Hybrid and staged CPU execution borrow a host context that must outlive this runtime.
    StagedMoeRuntime(const core::ExecutionContext & execution, StagedMoeConfig config,
        std::shared_ptr<core::GraphWorkspace> workspace, core::ExecutionContext * cpu_execution = nullptr);
    ~StagedMoeRuntime();
    StagedMoeRuntime(const StagedMoeRuntime &) = delete;
    StagedMoeRuntime & operator=(const StagedMoeRuntime &) = delete;

    // Bindings are borrowed and must outlive this runtime. CPU-bound caches on
    // a different backend execute synchronously on the borrowed CPU context;
    // whole-layer CPU placement takes precedence over hybrid sharing.
    size_t add_layer(const MoeModule & module, const MoeExpertWeights & weights,
        core::DiskExpertWeightStore & store, size_t group);
    // The caller prepares the primary workspace after registering every GPU graph.
    // CPU-only layers register no primary graphs; do not prepare an empty workspace.
    // Prepares execution plans and the separately owned CPU graph workspace;
    // run() may allocate host routing/cache temporaries.
    void prepare();
    void build_inputs(core::ModuleBuildContext & ctx, ggml_cgraph * graph,
        ggml_tensor * hidden, const MoeRouting & routing) const;
    // May be reused by a final backbone graph after expert execution completes.
    ggml_tensor * hidden_input() const;
    // Output lanes are restored to [hidden, selected position, token].
    ggml_tensor * output() const;
    ggml_tensor * routing_weights() const;
    // Returns padded execution rows, for existing caller timing/statistics.
    uint64_t run(size_t layer);
    // A graph-region executor already has host routing IDs.
    // Avoid uploading and reading the same IDs back through the device.
    uint64_t run(size_t layer, const std::vector<int32_t> & ids);
    size_t workspace_bytes() const;
    Statistics statistics() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
