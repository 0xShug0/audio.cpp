#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::miocodec {

struct MioCodecConfig;

using MioCodecLinearWeights = engine::modules::LinearWeights;

struct MioCodecConv1dWeights {
    engine::modules::Conv1dConfig config;
    engine::modules::Conv1dWeights weights;
};

struct MioCodecConvTranspose1dWeights {
    engine::modules::ConvTranspose1dConfig config;
    engine::modules::ConvTranspose1dWeights weights;
};

using MioCodecDepthwiseConv1dWeights = engine::modules::DepthwiseConv1dWeights;
using MioCodecNormWeights = engine::modules::NormWeights;

struct MioCodecAdaLayerNormWeights {
    MioCodecLinearWeights condition_projection;
};

struct MioCodecTransformerLayerWeights {
    MioCodecLinearWeights qkv_proj;
    MioCodecLinearWeights out_proj;
    MioCodecLinearWeights feed_forward_w1;
    MioCodecLinearWeights feed_forward_w2;
    MioCodecLinearWeights feed_forward_w3;
    std::optional<MioCodecNormWeights> attention_norm;
    std::optional<MioCodecNormWeights> feed_forward_norm;
    std::optional<MioCodecAdaLayerNormWeights> attention_adaln;
    std::optional<MioCodecAdaLayerNormWeights> feed_forward_adaln;
};

struct MioCodecTransformerWeights {
    int64_t dim = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t window_size = 0;
    int64_t intermediate_dim = 0;
    bool use_adaln = false;
    std::vector<MioCodecTransformerLayerWeights> layers;
    std::optional<MioCodecNormWeights> norm;
    std::optional<MioCodecAdaLayerNormWeights> adaln_norm;
    std::optional<MioCodecLinearWeights> output_projection;
};

struct MioCodecQuantizerWeights {
    MioCodecLinearWeights input_projection;
    MioCodecLinearWeights output_projection;
};

struct MioCodecConvNeXtBlockWeights {
    engine::modules::DepthwiseConv1dConfig depthwise_conv_config;
    MioCodecDepthwiseConv1dWeights depthwise_conv;
    MioCodecNormWeights norm;
    MioCodecLinearWeights pointwise_conv1;
    MioCodecLinearWeights pointwise_conv2;
    engine::core::TensorValue gamma;
};

struct MioCodecGlobalEncoderWeights {
    MioCodecConv1dWeights embedding;
    MioCodecNormWeights embedding_norm;
    std::vector<MioCodecConvNeXtBlockWeights> blocks;
    MioCodecNormWeights final_norm;
    MioCodecConv1dWeights attention_conv1;
    MioCodecConv1dWeights attention_conv2;
    MioCodecLinearWeights pooling_projection;
    MioCodecNormWeights pooling_norm;
};

struct MioCodecResNetBlockWeights {
    MioCodecNormWeights norm1;
    MioCodecConv1dWeights conv1;
    MioCodecNormWeights norm2;
    MioCodecConv1dWeights conv2;
};

struct MioCodecResNetStackWeights {
    std::vector<MioCodecResNetBlockWeights> blocks;
};

struct MioCodecSnakeBetaWeights {
    engine::core::TensorValue alpha;
    engine::core::TensorValue inv_beta;
};

struct MioCodecUpsamplerStageWeights {
    MioCodecConvTranspose1dWeights upsample;
    MioCodecSnakeBetaWeights snake;
    MioCodecResNetBlockWeights resnet;
};

struct MioCodecUpsamplerWeights {
    std::vector<MioCodecUpsamplerStageWeights> stages;
    MioCodecLinearWeights output_projection;
    MioCodecSnakeBetaWeights output_snake;
};

struct MioCodecIstftHeadWeights {
    MioCodecLinearWeights output_projection;
};

struct MioCodecContentTokenEmbeddingTable {
    std::vector<float> values;
    int64_t codebook_size = 0;
    int64_t dim = 768;
};

struct MioCodecWeights {
    std::filesystem::path source_path;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::unordered_map<std::string, engine::core::TensorValue> tensors;
    MioCodecTransformerWeights local_encoder;
    MioCodecQuantizerWeights local_quantizer;
    MioCodecConv1dWeights conv_downsample;
    MioCodecGlobalEncoderWeights global_encoder;
    MioCodecTransformerWeights wave_prenet;
    MioCodecConvTranspose1dWeights wave_conv_upsample;
    MioCodecResNetStackWeights wave_prior_net;
    MioCodecTransformerWeights wave_decoder;
    MioCodecResNetStackWeights wave_post_net;
    MioCodecUpsamplerWeights wave_upsampler;
    MioCodecIstftHeadWeights istft_head;
    MioCodecContentTokenEmbeddingTable content_token_embeddings;
    int64_t loaded_tensor_count = 0;
    int64_t parameter_count = 0;
};

std::shared_ptr<const MioCodecWeights> load_miocodec_weights(
    const engine::assets::TensorSource & source,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    const MioCodecConfig & config,
    engine::assets::TensorStorageType storage_type = engine::assets::TensorStorageType::F32);

}  // namespace engine::models::miocodec
