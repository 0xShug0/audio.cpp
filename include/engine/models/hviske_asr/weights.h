#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/hviske_asr/assets.h"

#include <memory>
#include <vector>

namespace engine::models::hviske_asr {

struct HviskeBatchNormEvalWeights {
    engine::core::TensorValue scale;
    engine::core::TensorValue bias;
};

struct HviskeSubsamplingWeights {
    engine::modules::Conv2dWeights conv0;
    engine::core::TensorValue depthwise1_weight;
    engine::core::TensorValue depthwise1_bias;
    engine::modules::Conv2dWeights pointwise1;
    engine::core::TensorValue depthwise2_weight;
    engine::core::TensorValue depthwise2_bias;
    engine::modules::Conv2dWeights pointwise2;
    engine::modules::LinearWeights linear;
};

struct HviskeEncoderLayerWeights {
    engine::modules::NormWeights norm_feed_forward1;
    engine::modules::NormWeights norm_self_att;
    engine::modules::NormWeights norm_conv;
    engine::modules::NormWeights norm_feed_forward2;
    engine::modules::NormWeights norm_out;
    engine::modules::LinearWeights ff1_linear1;
    engine::modules::LinearWeights ff1_linear2;
    engine::modules::LinearWeights ff2_linear1;
    engine::modules::LinearWeights ff2_linear2;
    engine::modules::RelativeAttentionWeights self_attn;
    engine::modules::LinearWeights conv_pointwise1;
    engine::modules::DepthwiseConv1dWeights conv_depthwise;
    HviskeBatchNormEvalWeights conv_norm;
    engine::modules::LinearWeights conv_pointwise2;
};

struct HviskeEncoderWeights {
    HviskeSubsamplingWeights subsampling;
    std::vector<HviskeEncoderLayerWeights> layers;
    engine::modules::LinearWeights encoder_projector;
};

struct HviskeDecoderLayerWeights {
    engine::modules::NormWeights self_norm;
    engine::modules::AttentionWeights self_attn;
    engine::modules::NormWeights cross_norm;
    engine::modules::AttentionWeights cross_attn;
    engine::modules::NormWeights ff_norm;
    engine::modules::LinearWeights ff_in;
    engine::modules::LinearWeights ff_out;
};

struct HviskeDecoderWeights {
    engine::core::TensorValue token_embedding;
    engine::core::TensorValue position_embedding;
    engine::modules::NormWeights embedding_norm;
    std::vector<HviskeDecoderLayerWeights> layers;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights classifier;
};

struct HviskeWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    HviskeEncoderWeights encoder;
    HviskeDecoderWeights decoder;
};

std::shared_ptr<const HviskeWeights> load_hviske_weights(
    const HviskeAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::models::hviske_asr
