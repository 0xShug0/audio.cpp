#include "engine/framework/neon/blocks.h"

namespace engine::neon {

ExecutionPlan::Value build_residual_conv(ExecutionPlan &plan, ExecutionPlan::Value input, int channels, int hidden,
                                         int kernel, ActivationKind activation, const ResidualConvWeights &weights) {
    const Conv1dConfig first{channels, hidden, kernel, 1, 1, kernel / 2, kernel / 2};
    const Conv1dConfig second{hidden, channels, kernel, 1, 1, kernel / 2, kernel / 2};
    auto value = plan.add(Conv1d(first, weights.first_weight, weights.first_bias), {input});
    value = plan.add(Activation(activation), {value});
    value = plan.add(Conv1d(second, weights.second_weight, weights.second_bias), {value});
    return plan.add(ScaleAdd(weights.scale), {input, value});
}

ExecutionPlan::Value build_convnext(ExecutionPlan &plan, ExecutionPlan::Value input, int channels, int expanded,
                                    int kernel, float epsilon, const ConvNeXtWeights &weights) {
    auto value = plan.add(DepthwiseConv1d(channels, kernel, weights.depthwise_weight, weights.depthwise_bias), {input});
    value = plan.add(LayerNorm(weights.norm_weight, weights.norm_bias, epsilon), {value});
    value = plan.add(Conv1d({channels, expanded, 1}, weights.expand_weight, weights.expand_bias), {value});
    value = plan.add(Activation(ActivationKind::GELU), {value});
    value = plan.add(Conv1d({expanded, channels, 1}, weights.project_weight, weights.project_bias), {value});
    return plan.add(ScaleAdd(weights.scale), {input, value});
}

} // namespace engine::neon
