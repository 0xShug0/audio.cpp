#include "engine/framework/modules/moe_feed_forward_module.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"

#include <stdexcept>
#include <utility>

namespace engine::modules {
namespace {

const core::ModulePortSpec kInputs[] = {{"input", core::PortKind::Activation, false}};
const core::ModulePortSpec kOutputs[] = {{"output", core::PortKind::Activation, false}};
const core::ModuleSchema kSchema = {
    "MoeFeedForward", "nn.block", kInputs, 1, kOutputs, 1,
    "Routed expert FFN with optional shared experts and latent projections.",
};

}  // namespace

MoeFeedForwardModule::MoeFeedForwardModule(MoeFeedForwardConfig config) : config_(std::move(config)) {
    // Validate expert arithmetic even when the caller uses staged execution.
    (void) MoeModule(config_.experts);
}

const MoeFeedForwardConfig & MoeFeedForwardModule::config() const noexcept { return config_; }
const core::ModuleSchema & MoeFeedForwardModule::schema() const noexcept { return static_schema(); }
const core::ModuleSchema & MoeFeedForwardModule::static_schema() noexcept { return kSchema; }

MoeFeedForwardInputs MoeFeedForwardModule::build_inputs(core::ModuleBuildContext & ctx,
    ggml_tensor * input, const MoeFeedForwardWeights & weights, const MoeRouting * routing) const {
    MoeFeedForwardInputs out;
    const auto shape = core::TensorShape::from_dims({input->ne[1], input->ne[0]});
    const auto x = core::wrap_tensor(input, shape);
    if (config_.shared) {
        core::TensorValue shared;
        if (const auto * config = std::get_if<FeedForwardConfig>(&*config_.shared)) {
            shared = FeedForwardModule(*config).build(ctx, x, std::get<FeedForwardWeights>(weights.shared));
        } else {
            shared = GatedFeedForwardModule(std::get<GatedFeedForwardConfig>(*config_.shared)).build(ctx, x,
                std::get<GatedFeedForwardWeights>(weights.shared));
        }
        out.shared = shared.tensor;
        if (weights.shared_gate.weight.valid()) {
            auto gate = LinearModule({input->ne[0], 1, weights.shared_gate.bias.has_value(), config_.projection_precision})
                .build(ctx, x, weights.shared_gate);
            gate = SigmoidModule{}.build(ctx, gate);
            out.shared = ggml_mul(ctx.ggml, out.shared, gate.tensor);
        }
    }
    auto router_input = x;
    if (!routing && config_.router_norm) router_input = RMSNormModule(*config_.router_norm).build(ctx, x, weights.router_norm);
    out.routing = routing ? *routing : MoeModule::build_routing(ctx, router_input.tensor,
        weights.router.tensor, weights.selection_bias.tensor, config_.routing);
    out.experts = input;
    if (weights.latent_down.weight.valid()) {
        const auto width = weights.latent_down.weight.tensor->ne[1];
        out.experts = LinearModule({input->ne[0], width, weights.latent_down.bias.has_value(), config_.projection_precision})
            .build(ctx, x, weights.latent_down).tensor;
    }
    return out;
}

ggml_tensor * MoeFeedForwardModule::finish(core::ModuleBuildContext & ctx, ggml_tensor * routed,
    ggml_tensor * shared, const MoeFeedForwardWeights & weights) const {
    if (config_.round_routed_output) routed = ggml_round_bf16(ctx.ggml, routed);
    if (weights.latent_up.weight.valid()) {
        auto input = core::wrap_tensor(routed, core::TensorShape::from_dims({routed->ne[1], routed->ne[0]}));
        routed = LinearModule({routed->ne[0], weights.latent_up.weight.tensor->ne[1],
            weights.latent_up.bias.has_value(), config_.projection_precision}).build(ctx, input, weights.latent_up).tensor;
    }
    if (config_.shared.has_value() != (shared != nullptr)) {
        throw std::invalid_argument("MoE FFN shared branch does not match its configuration");
    }
    if (shared) routed = ggml_add(ctx.ggml, routed, shared);
    return config_.round_output ? ggml_round_bf16(ctx.ggml, routed) : routed;
}

ggml_tensor * MoeFeedForwardModule::build_output(core::ModuleBuildContext & ctx,
    const MoeFeedForwardInputs & inputs, const MoeFeedForwardWeights & weights, ggml_tensor * bank_ids) const {
    auto * routed = MoeModule(config_.experts).build(ctx, inputs.experts, inputs.routing, weights.experts, bank_ids);
    return finish(ctx, routed, inputs.shared, weights);
}

ggml_tensor * MoeFeedForwardModule::build(core::ModuleBuildContext & ctx, ggml_tensor * input,
    const MoeFeedForwardWeights & weights, ggml_tensor * bank_ids, const MoeRouting * routing) const {
    return build_output(ctx, build_inputs(ctx, input, weights, routing), weights, bank_ids);
}

}  // namespace engine::modules
