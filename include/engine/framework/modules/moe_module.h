#pragma once

#include "engine/framework/core/module.h"

namespace engine::modules {

enum class MoeScoring { Softmax, Sigmoid };
enum class MoeNormalization { None, Divide, Clamp, AddEpsilon };
enum class MoeActivation { SwiGlu, ReluSquared };

struct MoeRoutingConfig {
    int64_t experts = 0;
    int64_t selected = 0;
    MoeScoring scoring = MoeScoring::Softmax;
    MoeNormalization normalization = MoeNormalization::Divide;
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

struct MoeExpertConfig {
    MoeActivation activation = MoeActivation::SwiGlu;
    bool fused_swiglu = false;
    // Boundaries are independently specified; storage type does not imply rounding.
    bool round_projections = false;
    bool round_activation = false;
    bool round_product = false;
    bool round_output = false;
};

struct MoeRouting {
    ggml_tensor * ids = nullptr;     // I32 [selected, tokens], logical expert IDs
    ggml_tensor * weights = nullptr; // [selected, tokens], in the same assignment order
};

struct MoeExpertWeights {
    ggml_tensor * up = nullptr;     // [physical input, intermediate, experts/slots]
    ggml_tensor * gate = nullptr;   // optional for ReLU squared
    ggml_tensor * down = nullptr;   // [physical intermediate, output, experts/slots]
};

// Graph construction only. The caller owns contexts, allocations and execution.
class MoeModule {
public:
    explicit MoeModule(MoeExpertConfig experts = {});
    static MoeRouting build_routing(core::ModuleBuildContext & ctx, ggml_tensor * input,
        ggml_tensor * router, ggml_tensor * selection_bias, const MoeRoutingConfig & config);
    // bank_ids index the supplied bank: logical IDs for resident weights, slot
    // IDs for staged weights. This never changes MoeRouting logical IDs.
    ggml_tensor * build_experts(core::ModuleBuildContext & ctx, ggml_tensor * input,
        ggml_tensor * bank_ids, const MoeExpertWeights & weights) const;
    static ggml_tensor * combine(core::ModuleBuildContext & ctx, ggml_tensor * outputs,
        ggml_tensor * routing_weights);
    // Explicit bank IDs allow a scheduler to stage a small routed batch inside
    // the caller graph without replacing the logical routing assignments.
    ggml_tensor * build(core::ModuleBuildContext & ctx, ggml_tensor * input,
        const MoeRouting & routing, const MoeExpertWeights & weights, ggml_tensor * bank_ids = nullptr) const;
private:
    MoeExpertConfig experts_;
};

} // namespace engine::modules
