#pragma once

#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <vector>

namespace engine::models::gigaam_asr {

struct ConformerConfig {
    int64_t features = 64;
    int64_t hidden_size = 768;
    int64_t intermediate_size = 3072;
    int64_t heads = 16;
    int64_t conv_kernel = 5;
    int64_t subsampling_kernel = 5;
    float rope_theta = 5000.0f;
    modules::ScaledDotProductAttentionLowering attention =
        modules::ScaledDotProductAttentionLowering::Flash;
};

struct ConformerLayerWeights {
    modules::NormWeights ffn1_norm, attention_norm, conv_norm, ffn2_norm, output_norm;
    modules::LinearWeights ffn1_in, ffn1_out, ffn2_in, ffn2_out;
    modules::LinearWeights query, key, value, attention_out;
    modules::LinearWeights pointwise_in, pointwise_out;
    modules::DepthwiseConv1dWeights depthwise;
    modules::NormWeights depthwise_norm;
};

struct ConformerWeights {
    modules::Conv1dWeights subsampling1, subsampling2;
    std::vector<ConformerLayerWeights> layers;
};

// Features are [1, mel_bins, frames]. Positions index the subsampled frames.
core::TensorValue build_conformer_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & features,
    const core::TensorValue & positions,
    const ConformerConfig & config,
    const ConformerWeights & weights);

}  // namespace engine::models::gigaam_asr
