#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/graph_workspace.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/modules/moe_module.h"

#include <memory>
#include <vector>

namespace engine::modules {

struct StagedMoeConfig {
    int64_t hidden_size = 0;
    int64_t tokens = 0;
    int64_t selected = 0;
    int64_t experts = 0;
    int64_t cache_slots = 0;
    // Execution buckets, independent of model identity. Preserve caller lowering.
    std::vector<int64_t> rows_per_expert;
    ggml_type scatter_index_type = GGML_TYPE_I64;
};

// Session-owned staged expert execution. Routing and surrounding backbone graphs
// are owned by the caller. Expert graph resources live here; activation scratch
// can be shared with serial backbone phases.
class StagedMoeRuntime {
public:
    StagedMoeRuntime(core::ExecutionContext & execution, StagedMoeConfig config,
        std::shared_ptr<core::GraphWorkspace> workspace);
    ~StagedMoeRuntime();
    StagedMoeRuntime(const StagedMoeRuntime &) = delete;
    StagedMoeRuntime & operator=(const StagedMoeRuntime &) = delete;

    // Bindings are borrowed and must outlive this runtime.
    size_t add_layer(const MoeModule & module, const MoeExpertWeights & weights,
        core::DiskExpertWeightStore & store, size_t group);
    // The caller prepares the shared workspace after registering every graph.
    // This prepares execution plans only; run() may allocate host temporaries.
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
    size_t workspace_bytes() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
