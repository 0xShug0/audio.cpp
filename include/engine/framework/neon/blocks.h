#pragma once

#include "engine/framework/neon/execution.h"

namespace engine::neon {

struct ResidualConvWeights {
    std::vector<float> first_weight, first_bias, second_weight, second_bias, scale;
};

struct ConvNeXtWeights {
    std::vector<float> depthwise_weight, depthwise_bias, norm_weight, norm_bias;
    std::vector<float> expand_weight, expand_bias, project_weight, project_bias, scale;
};

// These blocks compose ordinary operators and share the plan's allocation and
// execution policies. They do not introduce separate model-shaped executors.
ExecutionPlan::Value build_residual_conv(ExecutionPlan &plan, ExecutionPlan::Value input, int channels,
                                         int hidden_channels, int kernel_size, ActivationKind activation,
                                         const ResidualConvWeights &weights);

ExecutionPlan::Value build_convnext(ExecutionPlan &plan, ExecutionPlan::Value input, int channels,
                                    int expanded_channels, int kernel_size, float epsilon,
                                    const ConvNeXtWeights &weights);

} // namespace engine::neon
