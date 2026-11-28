#include "engine/framework/modules/convnext_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

namespace engine::modules {
core::TensorValue build_convnext1d(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const ConvNeXt1dWeights & weights, bool bf16_autocast) {
    core::validate_rank_between(input, 3, 3, "ConvNeXt1d input");
    const int64_t channels = input.shape.last_dim();
    const int64_t kernel = weights.depthwise.weight.shape.last_dim();
    const int64_t hidden = weights.expansion.weight.shape.at(0);
    const auto round = [&](core::TensorValue value) {
        if (!bf16_autocast) return value;
        return Bf16RoundModule().build(ctx, value);
    };
    auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, round(input));
    x = DepthwiseConv1dModule({channels, kernel, 1, int(kernel / 2), 1, true}).build(ctx, x, weights.depthwise);
    x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, round(x));
    x = LayerNormModule({channels, 1e-6f, true, true}).build(ctx, x, weights.norm);
    x = round(LinearModule({channels, hidden, true}).build(ctx, round(x), weights.expansion));
    x = round(GeluModule({GeluApproximation::ExactErf}).build(ctx, x));
    if (weights.response_gamma.valid()) {
        const auto energy = MulModule().build(ctx, x, x);
        const auto response = SqrtModule().build(ctx, ReduceSumModule({1}).build(ctx, energy));
        const auto mean = ScaleBiasModule({1, 1e-6f}).build(ctx, ReduceMeanModule({2}).build(ctx, response));
        const auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, response.tensor, mean.tensor), response.shape);
        const auto gamma = core::reshape_tensor(ctx, weights.response_gamma, core::TensorShape::from_dims({1, 1, hidden}));
        const auto beta = core::reshape_tensor(ctx, weights.response_beta, core::TensorShape::from_dims({1, 1, hidden}));
        auto scaled = MulModule(true).build(ctx, MulModule(true).build(ctx, x, normalized), gamma);
        scaled = AddModule(false, true).build(ctx, scaled, beta);
        x = AddModule().build(ctx, x, scaled);
    }
    x = round(LinearModule({hidden, channels, true}).build(ctx, round(x), weights.projection));
    if (!weights.response_gamma.valid() || weights.gamma.valid()) x = MulModule().build(ctx, x, RepeatModule({x.shape}).build(ctx, core::reshape_tensor(ctx, weights.gamma, core::TensorShape::from_dims({1, 1, channels}))));
    return AddModule().build(ctx, input, x);
}
}
