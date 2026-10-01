#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"

#include <array>

namespace engine::models::kugelaudio {

inline constexpr std::array<int32_t, 4> kSpeechTokens = {151652, 151653, 151654, 151643};

struct ArConfig {
    int64_t hidden_size = 3584;
    int64_t intermediate_size = 18944;
    int64_t attention_heads = 28;
    int64_t kv_heads = 4;
    int64_t head_dim = 128;
    int64_t layers = 28;
    int64_t vocab_size = 152064;
    int64_t max_position_embeddings = 32768;
    float norm_eps = 1e-6F;
    float rope_theta = 1000000.0F;
};

modules::CausalDecoderRuntimeConfig ar_runtime_config(const ArConfig & config);
modules::CausalDecoderRuntimeWeights load_ar_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    assets::TensorStorageType storage, const ArConfig & config);

}  // namespace engine::models::kugelaudio
