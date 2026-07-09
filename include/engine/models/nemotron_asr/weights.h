#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/nemotron_asr/assets.h"

#include <memory>
#include <vector>

namespace engine::models::nemotron_asr {

struct NemotronSubsamplingLayerWeights {
    engine::core::TensorValue depthwise_weight;
    engine::core::TensorValue depthwise_bias;
    engine::modules::Conv2dWeights pointwise;
};

struct NemotronSubsamplingWeights {
    engine::modules::Conv2dWeights conv_in;
    std::vector<NemotronSubsamplingLayerWeights> layers;
    engine::modules::LinearWeights linear;
};

struct NemotronEncoderLayerWeights {
    engine::modules::NormWeights norm_feed_forward1;
    engine::modules::NormWeights norm_self_att;
    engine::modules::NormWeights norm_conv;
    engine::modules::NormWeights conv_norm;
    engine::modules::NormWeights norm_feed_forward2;
    engine::modules::NormWeights norm_out;
    engine::modules::LinearWeights ff1_linear1;
    engine::modules::LinearWeights ff1_linear2;
    engine::modules::LinearWeights ff2_linear1;
    engine::modules::LinearWeights ff2_linear2;
    engine::modules::RelativeAttentionWeights self_attn;
    engine::modules::LinearWeights conv_pointwise1;
    engine::modules::DepthwiseConv1dWeights conv_depthwise;
    engine::modules::LinearWeights conv_pointwise2;
};

struct NemotronEncoderWeights {
    NemotronSubsamplingWeights subsampling;
    std::vector<NemotronEncoderLayerWeights> layers;
    engine::modules::LinearWeights prompt_linear1;
    engine::modules::LinearWeights prompt_linear2;
    engine::modules::LinearWeights encoder_projector;
};

struct NemotronDecoderWeights {
    engine::core::TensorValue embedding;
    std::vector<engine::modules::LSTMCellWeights> lstm_layers;
    engine::modules::LinearWeights decoder_projector;
    engine::modules::LinearWeights joint_head;
};

struct NemotronWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    NemotronEncoderWeights encoder;
    NemotronDecoderWeights decoder;
};

std::shared_ptr<const NemotronWeights> load_nemotron_asr_weights(
    const NemotronAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::models::nemotron_asr
