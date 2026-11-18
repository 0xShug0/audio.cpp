#pragma once

#include "engine/framework/moe_engine/moe_engine.h"
#include "engine/framework/core/prepared_graph.h"
#include <filesystem>
#include <functional>
#include <optional>

namespace engine::moe::detail {

struct ExpertWeights {
    ggml_tensor * up = nullptr;     // [physical input, intermediate, experts/slots]
    ggml_tensor * gate = nullptr;   // optional for ReLU squared
    ggml_tensor * down = nullptr;   // [physical intermediate, output, experts/slots]
};

Routing build_routing(core::ModuleBuildContext &, ggml_tensor *, ggml_tensor *, ggml_tensor *, const RoutingConfig &);
ggml_tensor * build_experts(core::ModuleBuildContext &, ggml_tensor *, ggml_tensor *, const ExpertWeights &, const ExpertMath &, ggml_tensor ** first = nullptr);
ggml_tensor * combine(core::ModuleBuildContext &, ggml_tensor *, ggml_tensor *);

class ExpertStorage {
public:
    struct Statistics {
        uint64_t bytes_read = 0;
        uint64_t expert_loads = 0;
        uint64_t cache_hits = 0;
        size_t buffer_bytes = 0;
        size_t host_buffer_bytes = 0;
        double read_ms = 0;
        double upload_ms = 0;
        double stage_ms = 0;
    };

    // Borrow the pinned host-bank provider while groups are registered.
    ExpertStorage(const std::filesystem::path & gguf_path, bool pin_experts,
        size_t read_threads, ggml_backend_t host_buffer_backend,
        std::function<std::unique_ptr<EvictionPolicy>(size_t)> eviction);
    ~ExpertStorage();
    ExpertStorage(const ExpertStorage &) = delete;
    ExpertStorage & operator=(const ExpertStorage &) = delete;

    // Projections in a group share an expert axis and are evicted together.
    size_t add_group(const std::vector<std::string> & tensor_names,
                     ggml_backend_t backend, int64_t cache_slots);
    core::TensorValue tensor(size_t group, size_t projection) const;
    // Borrow the backend owning this bounded cache.
    ggml_backend_t group_backend(size_t group) const;
    // Borrow native host weights without copying or expanding the expert bank.
    // The returned tensor and its mapped/pinned storage live with this store.
    core::TensorValue host_tensor(size_t group, const ggml_tensor * device_tensor);
    // Residency inspection does not change eviction order or choose execution.
    const std::vector<int32_t> & cached_experts(size_t group) const;
    uint64_t cache_epoch() const;
    // Stage one bounded group and return cache-slot IDs in the caller's order.
    const std::vector<int32_t> & stage(size_t group, const std::vector<int32_t> & expert_ids);
    size_t expert_bytes(size_t group) const;
    void accessed(size_t group, int32_t expert, size_t assignments);
    void event(size_t group, const PolicyNotification &);
    void event(const PolicyNotification &);
    // Invalidate placement at an independent request boundary, retaining all allocations and statistics.
    void invalidate_cache();
    Statistics statistics() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};


struct ProgramShape {
    uint64_t identity = 0;
    bool resident = false;
    bool native_single_token = false;
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
class ExpertProgram {
public:
    struct Statistics {
        uint64_t cpu_assignments = 0;
        uint64_t gpu_assignments = 0;
        double cpu_ms = 0;
        double gpu_ms = 0;
    };
    // Hybrid and staged CPU execution borrow a host context that must outlive this runtime.
    ExpertProgram(const core::ExecutionContext & execution, ProgramShape config,
        std::shared_ptr<core::GraphWorkspace> workspace, core::ExecutionContext * cpu_execution, const PreparedPolicy & policy);
    ~ExpertProgram();
    ExpertProgram(const ExpertProgram &) = delete;
    ExpertProgram & operator=(const ExpertProgram &) = delete;

    // Bindings are borrowed and must outlive this runtime. CPU-bound caches on
    // a different backend execute synchronously on the borrowed CPU context;
    // whole-layer CPU placement takes precedence over hybrid sharing.
    size_t add_layer(const ExpertMath & module, const ExpertWeights & weights,
        ExpertStorage * store, size_t group, ggml_backend_t backend, PlacementPolicy * placement);
    // The caller prepares the primary workspace after registering every GPU graph.
    // CPU-only layers register no primary graphs; do not prepare an empty workspace.
    // Prepares execution plans and the separately owned CPU graph workspace;
    // Host routing/cache workspaces are reserved during preparation.
    void prepare();
    void build_inputs(core::ModuleBuildContext & ctx, ggml_cgraph * graph,
        ggml_tensor * hidden, const Routing & routing) const;
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
    bool needs_primary_workspace() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};


} // namespace engine::moe::detail
