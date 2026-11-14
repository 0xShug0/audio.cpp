#pragma once

#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/moe_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <optional>
#include <variant>

namespace engine::modules {

using SharedFeedForwardConfig = std::variant<FeedForwardConfig, GatedFeedForwardConfig>;
using SharedFeedForwardWeights = std::variant<FeedForwardWeights, GatedFeedForwardWeights>;

struct MoeFeedForwardConfig {
    MoeRoutingConfig routing;
    MoeExpertConfig experts;
    std::optional<SharedFeedForwardConfig> shared;
    std::optional<NormConfig> router_norm;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    bool round_routed_output = false;
    bool round_output = false;
};

struct MoeFeedForwardWeights {
    core::TensorValue router, selection_bias;
    MoeExpertWeights experts;
    NormWeights router_norm;
    SharedFeedForwardWeights shared{};
    LinearWeights shared_gate{};
    LinearWeights latent_down{}, latent_up{};
};

struct MoeFeedForwardInputs {
    ggml_tensor * experts = nullptr;
    MoeRouting routing;
    ggml_tensor * shared = nullptr;
};

// Stateless graph composition. Expert storage, placement and execution remain
// with MoeResources and the caller's runtime. The split build API expresses the
// same FFN across staged expert execution, without owning graphs or state.
class MoeFeedForwardModule {
public:
    explicit MoeFeedForwardModule(MoeFeedForwardConfig config);
    const MoeFeedForwardConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    static const core::ModuleSchema & static_schema() noexcept;

    // An externally computed route supports recurrent or nonlinear routers.
    MoeFeedForwardInputs build_inputs(core::ModuleBuildContext & ctx, ggml_tensor * input,
        const MoeFeedForwardWeights & weights, const MoeRouting * routing = nullptr) const;
    ggml_tensor * build_output(core::ModuleBuildContext & ctx, const MoeFeedForwardInputs & inputs,
        const MoeFeedForwardWeights & weights, ggml_tensor * bank_ids = nullptr) const;
    // routed is the weighted expert reduction, in original assignment order.
    ggml_tensor * finish(core::ModuleBuildContext & ctx, ggml_tensor * routed,
        ggml_tensor * shared, const MoeFeedForwardWeights & weights) const;
    ggml_tensor * build(core::ModuleBuildContext & ctx, ggml_tensor * input,
        const MoeFeedForwardWeights & weights, ggml_tensor * bank_ids = nullptr,
        const MoeRouting * routing = nullptr) const;

private:
    MoeFeedForwardConfig config_;
};

}  // namespace engine::modules
