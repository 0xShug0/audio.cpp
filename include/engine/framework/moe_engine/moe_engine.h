#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/graph_workspace.h"
#include "engine/framework/moe_engine/moe_policy.h"

#include <memory>
#include <vector>

namespace engine::moe {
class MoePreparedGraph;

enum class Scoring { Softmax, Sigmoid, SelectedSoftmax };
enum class Normalization { None, Divide, Clamp, AddEpsilon };
enum class ExpertActivation { SwiGlu, ReluSquared };

struct RoutingConfig {
    int64_t experts = 0;
    int64_t selected = 0;
    Scoring scoring = Scoring::Softmax;
    Normalization normalization = Normalization::Divide;
    float denominator_epsilon = 0.0F;
    float scale = 1.0F;
    ggml_prec precision = GGML_PREC_DEFAULT;
    int64_t groups = 1;
    int64_t selected_groups = 1;
    // Rank groups by the sum of their top-k selection scores (including bias).
    int64_t group_top_k = 2;
    // Materialize selected IDs before gathering weights, preserving that graph boundary.
    bool contiguous_ids = false;
};

struct ExpertMath {
    ExpertActivation activation = ExpertActivation::SwiGlu;
    bool fused_swiglu = false;
    // Boundaries are independently specified; storage type does not imply rounding.
    bool round_projections = false;
    bool round_activation = false;
    bool round_product = false;
    bool round_output = false;
};

struct Routing {
    ggml_tensor * ids = nullptr;     // I32 [selected, tokens], logical expert IDs
    ggml_tensor * weights = nullptr; // [selected, tokens], in the same assignment order
};

enum class Residency { Resident, Staged };
struct EngineConfig {
    Residency residency = Residency::Resident;
    bool pin_host_weights = false;
    int64_t cache_slots = 8;
    int64_t read_threads = 1;
    std::vector<size_t> cpu_layers;
    PolicyConfig policies;
};
enum class ProjectionRole { Up, Gate, Down };
struct ExpertTensorBinding {
    std::string name;
    std::string disk_name;
    std::vector<int64_t> shape;
    assets::TensorStorageType storage = assets::TensorStorageType::Native;
    ProjectionRole role = ProjectionRole::Up;
};
struct LayerBindings {
    core::TensorValue router, selection_bias;
    std::vector<ExpertTensorBinding> experts;
};
struct LayerMath {
    RoutingConfig routing;
    ExpertMath experts;
};
struct LayerId {
    size_t index = 0;
    const void * owner = nullptr;
};
struct CallId {
    size_t index = 0;
    const void * owner = nullptr;
    uint64_t generation = 0;
};
enum class ExpertLowering { Default, FusedSwiGlu };
struct CallShape {
    int64_t tokens = 0;
    int64_t input_width = 0;
    int64_t output_width = 0;
    int64_t maximum_rows = 32;
    ggml_type scatter_index_type = GGML_TYPE_I64;
    int64_t minimum_rows = 1;
    ExpertLowering lowering = ExpertLowering::Default;
    // Embedded one-token regions replace native top-k expert operations.
    // Phased execution preserves its expert-row buckets by default.
    bool native_single_token = false;
};
struct ExpertStatistics {
    uint64_t bytes_read = 0, expert_loads = 0, cache_hits = 0;
    size_t buffer_bytes = 0, host_buffer_bytes = 0;
    double read_ms = 0, upload_ms = 0, stage_ms = 0;
};

class MoeEngine {
public:
    MoeEngine(core::ExecutionContext & execution, core::BackendWeightStore & primary_weights,
        const assets::TensorSource & source, EngineConfig config,
        std::string label = "framework.moe.weights", size_t metadata_bytes = 16 * 1024 * 1024);
    ~MoeEngine();
    MoeEngine(const MoeEngine &) = delete;
    MoeEngine & operator=(const MoeEngine &) = delete;

    LayerId register_layer(LayerBindings bindings, LayerMath math);
    void finalize_weights();
    const EngineConfig & config() const noexcept;
    ExpertStatistics statistics() const noexcept;
    bool staged() const noexcept;
    void begin_generation(bool invalidate_weights = true);
    void release_execution();

    Routing route(core::ModuleBuildContext & ctx, LayerId layer, ggml_tensor * input) const;
    ggml_tensor * build(MoePreparedGraph & graph, LayerId layer, ggml_tensor * input,
        const Routing * routing = nullptr, ExpertLowering lowering = ExpertLowering::Default);
    CallId add_call(LayerId layer, CallShape shape, std::shared_ptr<core::GraphWorkspace> workspace);
    void build_inputs(CallId call, MoePreparedGraph & graph, size_t graph_id,
        ggml_tensor * input, const Routing * routing = nullptr);
    ggml_tensor * build_result(CallId call, MoePreparedGraph & graph);

private:
    friend class MoePreparedGraph;
    class Impl;
    std::unique_ptr<Impl> impl_;
    void acquire();
    void drain();
    void relinquish() noexcept;
    bool needs_primary_workspace(CallId call) const;
    void prepare(CallId call);
    uint64_t execute(CallId call, const std::vector<int32_t> * ids = nullptr);
    const std::vector<int32_t> & stage_routes(CallId call, const std::vector<int32_t> & ids);
    ggml_backend_t routing_backend();
    void release(CallId call);
    ggml_tensor * input_port(CallId call) const;
    ggml_tensor * output_port(CallId call) const;
    core::ExecutionContext & execution() const;
};
} // namespace engine::moe
