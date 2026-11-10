#pragma once

#include "engine/framework/core/prepared_graph.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace engine::modules {

struct FlowSamplerGraphStepBinding {
    core::TensorValue target;
    size_t offset_bytes = 0;
    size_t bytes_per_step = 0;
    // Packed native tensor bytes, one equally sized slice per step. No casts.
    const void * data = nullptr;
    size_t data_bytes = 0;
};

struct FlowSamplerGraphSequence {
    size_t steps = 0;
    std::vector<FlowSamplerGraphStepBinding> step_bindings;
    // Optional graph ID per submission. Empty selects graph zero.
    std::vector<core::PreparedGraph::GraphId> graph_indices;
};

// Borrows a session-owned graph whose update writes back into latent. Serial
// use only, on the graph's execution backend. No graph construction in run().
// The caller prepares conditioning and sequence constants before run(). The
// graph builder must preserve state and constants across submissions, using
// external storage or explicitly preserved graph storage. INPUT alone does not
// prevent allocator reuse. Storage ownership remains with the caller.
// Step bindings are overwritten before every submission and need not retain
// their contents afterward. All borrowed storage must outlive the sequence.
class DeviceFlowSamplerRuntime {
public:
    DeviceFlowSamplerRuntime(core::PreparedGraph & graph, core::TensorValue latent);

    // Caller keeps all source spans immutable and alive until run() returns or
    // throws: completion/error draining synchronizes before releasing the borrow.
    // Bindings may contain F32, I32 or other native bytes. Target ranges must not
    // overlap the latent or one another. No source payloads are copied or retained.
    std::vector<float> run(
        const std::vector<float> & initial_latent,
        const FlowSamplerGraphSequence & sequence);

private:
    core::PreparedGraph & graph_;
    core::TensorValue latent_;
};

enum class FlowSamplerGraphRounding {
    FullPrecision,
    Bf16,
};

struct FlowSamplerEulerGraphConfig {
    FlowSamplerGraphRounding delta_rounding = FlowSamplerGraphRounding::FullPrecision;
    FlowSamplerGraphRounding output_rounding = FlowSamplerGraphRounding::FullPrecision;
};

// Tensor expressions only: callers choose branch slices and explicitly append
// the persistent latent copy to their graph. Scalars are F32 one-element tensors.
class FlowSamplerCfgGraphModule {
public:
    core::TensorValue build(core::ModuleBuildContext & ctx,
        const core::TensorValue & positive, const core::TensorValue & negative,
        const core::TensorValue & guidance) const;
};

class FlowSamplerEulerGraphModule {
public:
    explicit FlowSamplerEulerGraphModule(FlowSamplerEulerGraphConfig config = {});
    core::TensorValue build(core::ModuleBuildContext & ctx,
        const core::TensorValue & latent, const core::TensorValue & velocity,
        const core::TensorValue & dt) const;

private:
    FlowSamplerEulerGraphConfig config_;
};

struct DeviceDenoiserGraph {
    // Sampling state and its update are F32; prediction keeps its graph dtype.
    core::TensorValue latent;
    core::TensorValue prediction;
    core::TensorValue updated_latent;
};

struct DeviceDenoiserOptimizations {
    bool norm_affine = true;
    bool bias_activation = true;
    bool gated_residual = true;
};

// Owns immutable denoiser phases and their device sampling loop. The builder
// supplies model math and borrowed weights/conditioning; those must outlive this
// runtime. Phases share persistent latent storage and serial scratch allocation.
// Multiphase state and constants shared between phases must be separately
// allocated before construction, outside this runtime's scratch workspace.
// Graph-allocated constants reused across submissions must be explicitly
// preserved with OUTPUT; INPUT alone permits allocator reuse. Use is serial on
// the execution backend. A sequence can select a phase for each submission.
// Construction opts this graph into supported backend optimizations. Other graphs
// keep ordinary dispatch; unsupported patterns retain their original operations.
class DeviceDenoiserRuntime {
public:
    using Builder = std::function<DeviceDenoiserGraph(core::ModuleBuildContext &)>;

    DeviceDenoiserRuntime(core::ExecutionContext & execution,
        core::PreparedGraphConfig config, const Builder & builder,
        DeviceDenoiserOptimizations optimizations = {});
    // One metadata capacity covers all phases; graph_nodes is the per-graph capacity.
    DeviceDenoiserRuntime(core::ExecutionContext & execution,
        core::PreparedGraphConfig config, const std::vector<Builder> & phases,
        DeviceDenoiserOptimizations optimizations = {});
    ~DeviceDenoiserRuntime();

    std::vector<float> predict(size_t phase = 0);
    std::vector<float> sample(const std::vector<float> & initial_latent,
        const FlowSamplerGraphSequence & sequence);

private:
    core::PreparedGraph graph_;
    std::vector<DeviceDenoiserGraph> tensors_;
    std::unique_ptr<DeviceFlowSamplerRuntime> sampler_;
    ggml_backend_set_graph_fusions_t set_fusions_ = nullptr;
};

}  // namespace engine::modules
