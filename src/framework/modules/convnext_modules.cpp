#include "engine/framework/modules/convnext_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

namespace engine::modules {
core::TensorValue build_convnext1d(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const ConvNeXt1dWeights & weights) {
    core::validate_rank_between(input, 3, 3, "ConvNeXt1d input");
    const int64_t channels = input.shape.last_dim();
    const int64_t kernel = weights.depthwise.weight.shape.last_dim();
    const int64_t hidden = weights.expansion.weight.shape.at(0);
    auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    x = DepthwiseConv1dModule({channels, kernel, 1, int(kernel / 2), 1, true}).build(ctx, x, weights.depthwise);
    x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = LayerNormModule({channels, 1e-6f, true, true}).build(ctx, x, weights.norm);
    x = LinearModule({channels, hidden, true}).build(ctx, x, weights.expansion);
    x = GeluModule({GeluApproximation::ExactErf}).build(ctx, x);
    x = LinearModule({hidden, channels, true}).build(ctx, x, weights.projection);
    x = MulModule().build(ctx, x, RepeatModule({x.shape}).build(ctx, core::reshape_tensor(ctx, weights.gamma, core::TensorShape::from_dims({1, 1, channels}))));
    return AddModule().build(ctx, input, x);
}
}
