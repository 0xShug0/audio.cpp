#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/models/moonshine_stt/assets.h"

#include <memory>
#include <optional>
#include <vector>

namespace engine::models::moonshine_stt {

struct MoonshineFrontendWeights {
    float log_k = 0.0F;
    engine::modules::LinearWeights linear;
    engine::modules::Conv1dWeights conv1;
    engine::modules::Conv1dWeights conv2;
};

struct MoonshineEncoderLayerWeights {
    engine::modules::NormWeights input_norm;
    engine::modules::AttentionWeights self_attn;
    engine::modules::NormWeights post_attention_norm;
    engine::modules::FeedForwardWeights feed_forward;
};

struct MoonshineEncoderWeights {
    MoonshineFrontendWeights frontend;
    std::vector<MoonshineEncoderLayerWeights> layers;
    engine::modules::NormWeights final_norm;
};

struct MoonshineDecoderLayerWeights {
    engine::modules::NormWeights input_norm;
    engine::modules::AttentionWeights self_attn;
    engine::modules::NormWeights post_attention_norm;
    engine::modules::AttentionWeights cross_attn;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights mlp_fc1;
    engine::modules::LinearWeights mlp_fc2;
};

struct MoonshineDecoderWeights {
    engine::core::TensorValue token_embedding;
    engine::core::TensorValue position_embedding;
    std::optional<engine::modules::LinearWeights> adapter_proj;
    std::vector<MoonshineDecoderLayerWeights> layers;
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights output_projection;
};

struct MoonshineWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    MoonshineEncoderWeights encoder;
    MoonshineDecoderWeights decoder;
};

std::shared_ptr<const MoonshineWeights> load_moonshine_stt_weights(
    const MoonshineAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType encoder_storage_type,
    engine::assets::TensorStorageType decoder_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::models::moonshine_stt
