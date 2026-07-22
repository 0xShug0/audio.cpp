#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::sortformer_diar {

struct SortformerFeatureExtractorConfig {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t win_length = 0;
    int64_t hop_length = 0;
    int64_t num_mel_bins = 0;
    float preemphasis = 0.0f;
    bool return_attention_mask = true;
};

struct BatchNorm1dEvalWeights {
    core::TensorValue scale;
    core::TensorValue bias;
};

struct SortformerSubsamplingWeights {
    modules::Conv2dWeights conv0;
    core::TensorValue depthwise1_weight;
    core::TensorValue depthwise1_bias;
    modules::Conv2dWeights pointwise1;
    core::TensorValue depthwise2_weight;
    core::TensorValue depthwise2_bias;
    modules::Conv2dWeights pointwise2;
    modules::LinearWeights linear;
};

struct SortformerConformerLayerWeights {
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

struct SortformerTransformerLayerWeights {
    modules::NormWeights self_attn_layer_norm;
    modules::LinearWeights self_attn_q_proj;
    modules::LinearWeights self_attn_k_proj;
    modules::LinearWeights self_attn_v_proj;
    modules::LinearWeights self_attn_out_proj;
    modules::NormWeights final_layer_norm;
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
};

struct SortformerHeadWeights {
    modules::LinearWeights encoder_proj;
    modules::LinearWeights first_hidden_to_hidden;
    modules::LinearWeights single_hidden_to_spks;
};

struct SortformerFastConformerConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_key_value_heads = 0;
    int64_t num_mel_bins = 0;
    int64_t max_position_embeddings = 0;
    int64_t conv_kernel_size = 0;
    int64_t subsampling_factor = 0;
    int64_t subsampling_conv_channels = 0;
    int64_t subsampling_conv_kernel_size = 0;
    int64_t subsampling_conv_stride = 0;
    bool attention_bias = false;
    bool scale_input = false;
    std::string hidden_act;
};

struct SortformerTransformerConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t max_source_positions = 0;
    float layer_norm_eps = 1.0e-5f;
    std::string activation_function;
};

struct SortformerModulesConfig {
    int64_t num_speakers = 0;
    int64_t fc_d_model = 0;
    int64_t tf_d_model = 0;
    int64_t subsampling_factor = 0;
    float dropout_rate = 0.0f;
};

struct SortformerModelConfig {
    std::string model_type;
    std::string variant;
    int64_t num_speakers = 0;
    float pil_weight = 0.0f;
    float ats_weight = 0.0f;
    SortformerFastConformerConfig fc_encoder;
    SortformerTransformerConfig tf_encoder;
    SortformerModulesConfig modules;
};

struct SortformerDiarWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    SortformerSubsamplingWeights subsampling;
    std::vector<SortformerConformerLayerWeights> conformer_layers;
    std::vector<SortformerTransformerLayerWeights> transformer_layers;
    SortformerHeadWeights head;
};

struct SortformerAssets {
    assets::ResourceBundle resources;
    SortformerModelConfig model_config;
    SortformerFeatureExtractorConfig feature_config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const SortformerAssets> load_sortformer_assets(const std::filesystem::path & model_root);
SortformerModelConfig parse_sortformer_model_config(const assets::ResourceBundle & resources);
SortformerFeatureExtractorConfig parse_sortformer_feature_config(const assets::ResourceBundle & resources);
std::shared_ptr<SortformerDiarWeights> load_sortformer_diar_weights(
    const SortformerAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::models::sortformer_diar
