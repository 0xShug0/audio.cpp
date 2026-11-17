#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/moe_engine/disk_expert_weight_store.h"
#include "engine/framework/moe_engine/moe_config.h"
#include "engine/framework/moe_engine/moe_module.h"
#include "engine/framework/moe_engine/placement_policy.h"

#include <memory>
#include <optional>

namespace engine::modules {

class MoeExecutionPlan;
struct MoeGraphState;
struct StagedMoeConfig;
class StagedMoeRuntime;

enum class MoeExpertProjection { Up, Gate, Down };

struct MoeExpertTensorBinding {
    MoeExpertProjection role;
    std::string name;       // Logical name in TensorSource.
    std::string disk_name;  // Logical name in the containing GGUF.
    std::vector<int64_t> shape;
    assets::TensorStorageType storage = assets::TensorStorageType::Native;
};

struct MoeExpertLayer {
    size_t ordinal = 0;
    size_t backbone_index = 0;
    bool cpu = false;
    int64_t experts = 0;
    int64_t selected = 0;
    std::vector<core::TensorValue> weights;
    MoeLayerPlacementPolicy placement;
    MoeExpertWeights projections;
    core::DiskExpertWeightStore * disk_store = nullptr;
    size_t disk_group = 0;
};

// Session-owned expert engine: weight registration, residency and policy state,
// plus shape-specific execution plans. Graph-only routing/shared-expert math
// stays composable through the existing modules. Primary weights/context must
// outlive the engine; borrowing graphs/plans must be destroyed before it.
class MoeEngine {
public:
    MoeEngine(core::ExecutionContext & primary, const assets::TensorSource & source,
        core::BackendWeightStore & primary_weights, core::MoeExecutionPolicy policy,
        std::string cpu_weight_label, size_t cpu_weight_context_bytes);
    ~MoeEngine();
    MoeEngine(const MoeEngine &) = delete;
    MoeEngine & operator=(const MoeEngine &) = delete;

    const MoeExpertLayer & bind_layer(size_t ordinal, size_t backbone_index,
        const std::vector<MoeExpertTensorBinding> & tensors, int64_t selected);
    // Finalize registration and upload auxiliary CPU banks. The caller uploads
    // the shared primary store in its original order before this call.
    void finalize_weights();
    const MoeExpertLayer & layer(size_t ordinal) const;
    const MoeExpertLayer & layer_for_group(size_t group) const;
    const core::MoeExecutionPolicy & policy() const noexcept;
    core::DiskExpertWeightStore * disk_store() const noexcept;
    core::ExecutionContext * cpu_execution() const noexcept;
    std::unique_ptr<MoeExecutionPlan> create_plan(std::shared_ptr<core::GraphWorkspace> workspace) const;
    bool has_cpu_layers() const noexcept;
    // Install the existing scheduler and disk evaluator only when needed.
    ggml_backend_sched_t configure_scheduler(core::PreparedGraph & graph) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Shape-dependent batching only. Residency, expert counts, selection width and
// cache capacity come from the registered engine layer, never a second policy.
struct MoeBatchConfig {
    int64_t hidden_size = 0;
    int64_t tokens = 0;
    std::vector<int64_t> rows_per_expert;
    ggml_type scatter_index_type = GGML_TYPE_I64;
    int64_t input_size = 0; // Zero uses hidden_size.
};

// Compiled bounded-expert execution for one backbone graph shape. Models build
// their backbone/state graphs; the engine owns expert shape reuse, preparation
// and expert execution. Models retain their backbone execution loops.
// All borrowed graphs and expert banks must outlive this plan.
class MoeExecutionPlan {
public:
    using LayerId = size_t;
    ~MoeExecutionPlan();
    MoeExecutionPlan(const MoeExecutionPlan &) = delete;
    MoeExecutionPlan & operator=(const MoeExecutionPlan &) = delete;

    // Each call keeps its existing arithmetic; projection roles and residency
    // come only from registration. No bank rebinding.
    LayerId add_layer(size_t ordinal, const MoeBatchConfig & shape, const MoeExpertConfig & config);
    void build_inputs(LayerId layer, core::ModuleBuildContext & ctx, ggml_cgraph * graph,
        ggml_tensor * hidden, const MoeRouting & routing) const;
    ggml_tensor * hidden_input(LayerId layer) const;
    ggml_tensor * output(LayerId layer) const;
    ggml_tensor * routing_weights(LayerId layer) const;

    // Call after the shared workspace has reserved every backbone/expert shape.
    // A failure invalidates this plan. Rebuild its graph owner and workspace.
    void prepare();
    uint64_t run(LayerId layer);
private:
    friend class MoeEngine;
    friend struct MoeGraphState;
    friend class MoeRegionExecutor;
    MoeExecutionPlan(const MoeEngine & engine, const core::ExecutionContext & execution,
        std::shared_ptr<core::GraphWorkspace> workspace, core::ExecutionContext * cpu_execution);
    StagedMoeRuntime & runtime(LayerId layer) const;
    size_t runtime_layer(LayerId layer) const;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
