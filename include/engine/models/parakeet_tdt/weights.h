#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/models/parakeet_tdt/assets.h"

#include <memory>
#include <vector>

namespace engine::models::parakeet_tdt {

struct BatchNorm1dEvalWeights {
    core::TensorValue scale;
    core::TensorValue bias;
};

struct ParakeetConvSubsamplingWeights {
    modules::Conv2dWeights conv0;
    core::TensorValue depthwise1_weight;
    core::TensorValue depthwise1_bias;
    modules::Conv2dWeights pointwise1;
    core::TensorValue depthwise2_weight;
    core::TensorValue depthwise2_bias;
    modules::Conv2dWeights pointwise2;
    modules::LinearWeights linear;
};

struct ParakeetEncoderLayerWeights {
    modules::NormWeights norm_feed_forward1;
    modules::NormWeights norm_self_att;
    modules::NormWeights norm_conv;
    modules::NormWeights norm_feed_forward2;
    modules::NormWeights norm_out;

    modules::LinearWeights ff1_linear1;
    modules::LinearWeights ff1_linear2;
    modules::LinearWeights ff2_linear1;
    modules::LinearWeights ff2_linear2;

    modules::RelativeAttentionWeights self_attn;

    modules::LinearWeights conv_pointwise_conv1;
    modules::DepthwiseConv1dWeights conv_depthwise_conv;
    BatchNorm1dEvalWeights conv_norm;
    modules::LinearWeights conv_pointwise_conv2;
};

struct ParakeetEncoderWeights {
    ParakeetConvSubsamplingWeights subsampling;
    std::vector<ParakeetEncoderLayerWeights> layers;
    modules::LinearWeights encoder_projector;
};

struct ParakeetDecoderWeights {
    core::TensorValue embedding_weight;
    modules::LSTMCellWeights lstm0;
    modules::LSTMCellWeights lstm1;
    modules::LinearWeights decoder_projector;
};

struct ParakeetJointWeights {
    modules::LinearWeights head;
};

struct ParakeetTDTWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    ParakeetEncoderWeights encoder;
    ParakeetDecoderWeights decoder;
    ParakeetJointWeights joint;
};

std::shared_ptr<const ParakeetTDTWeights> load_parakeet_tdt_weights(
    const ParakeetAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::models::parakeet_tdt
