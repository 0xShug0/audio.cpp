#pragma once

#include "engine/community_models/sortformer_diar/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::sortformer_diar {

struct SortformerV2BatchNormWeights {
    core::TensorValue scale;
    core::TensorValue bias;
};

struct SortformerV2SubsamplingWeights {
    modules::Conv2dWeights conv0;
    core::TensorValue depthwise1_weight;
    core::TensorValue depthwise1_bias;
    modules::Conv2dWeights pointwise1;
    core::TensorValue depthwise2_weight;
    core::TensorValue depthwise2_bias;
    modules::Conv2dWeights pointwise2;
    modules::LinearWeights linear;
};

struct SortformerV2ConformerLayerWeights {
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
    SortformerV2BatchNormWeights conv_norm;
    modules::LinearWeights conv_pointwise_conv2;
};

struct SortformerV2TransformerLayerWeights {
    modules::NormWeights self_attn_layer_norm;
    modules::LinearWeights self_attn_q_proj;
    modules::LinearWeights self_attn_k_proj;
    modules::LinearWeights self_attn_v_proj;
    modules::LinearWeights self_attn_out_proj;
    modules::NormWeights final_layer_norm;
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
};

struct SortformerV2HeadWeights {
    modules::LinearWeights encoder_proj;
    modules::LinearWeights first_hidden_to_hidden;
    modules::LinearWeights single_hidden_to_spks;
};

struct SortformerV2Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    SortformerV2SubsamplingWeights subsampling;
    std::vector<SortformerV2ConformerLayerWeights> conformer_layers;
    std::vector<SortformerV2TransformerLayerWeights> transformer_layers;
    SortformerV2HeadWeights head;
};

std::shared_ptr<const SortformerV2Weights> load_sortformer_v2_weights(
    const SortformerV2Assets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::community_models::sortformer_diar
