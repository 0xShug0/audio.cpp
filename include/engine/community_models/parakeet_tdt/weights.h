#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/community_models/parakeet_tdt/assets.h"

#include <memory>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct ParakeetSubsamplingLayerWeights {
    engine::core::TensorValue depthwise_weight;
    engine::core::TensorValue depthwise_bias;
    engine::modules::Conv2dWeights pointwise;
};

struct ParakeetSubsamplingWeights {
    engine::modules::Conv2dWeights conv_in;
    std::vector<ParakeetSubsamplingLayerWeights> layers;
    engine::modules::LinearWeights linear;
};

struct ParakeetEncoderLayerWeights {
    engine::modules::NormWeights norm_ff1;
    engine::modules::LinearWeights ff1_linear1;
    engine::modules::LinearWeights ff1_linear2;
    engine::modules::NormWeights norm_attn;
    engine::modules::AttentionWeights self_attn;
    engine::core::TensorValue pos_weight;
    engine::core::TensorValue pos_bias_u;
    engine::core::TensorValue pos_bias_v;
    engine::modules::NormWeights norm_conv;
    engine::modules::LinearWeights conv_pw1;
    engine::core::TensorValue conv_dw_weight;
    engine::core::TensorValue conv_dw_bias;
    engine::modules::LinearWeights conv_pw2;
    engine::modules::NormWeights norm_ff2;
    engine::modules::LinearWeights ff2_linear1;
    engine::modules::LinearWeights ff2_linear2;
    engine::modules::NormWeights norm_out;
};

struct ParakeetEncoderWeights {
    ParakeetSubsamplingWeights subsampling;
    std::vector<ParakeetEncoderLayerWeights> layers;
};

struct ParakeetDecoderWeights {
    engine::core::TensorValue embedding;
    std::vector<engine::modules::LSTMCellWeights> lstm_layers;
    engine::modules::LinearWeights decoder_projector;
    engine::modules::LinearWeights joint_enc;    // encoder_projector: 1024→640
    engine::modules::LinearWeights joint_head;   // 640→8198
};

struct ParakeetWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    ParakeetEncoderWeights encoder;
    ParakeetDecoderWeights decoder;
};

std::shared_ptr<const ParakeetWeights> load_parakeet_weights(
    const ParakeetTDTAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::community_models::parakeet_tdt
