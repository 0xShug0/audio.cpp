#pragma once

#include "engine/framework/core/module.h"

#include <optional>

namespace engine::modules {

enum class AttentionPrefixCacheLayout {
    SequenceHeads,
    HeadsSequence,
};

struct AttentionConfig {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    bool use_bias = true;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
    AttentionPrefixCacheLayout prefix_cache_layout = AttentionPrefixCacheLayout::SequenceHeads;
};

struct RelativeAttentionConfig {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    bool use_bias = true;
    int64_t left_context = -1;
    int64_t right_context = -1;
    int64_t cache_drop_size = 0;
    bool use_flash_attention = false;
};

struct AttentionWeights {
    core::TensorValue q_weight;
    std::optional<core::TensorValue> q_bias;
    core::TensorValue k_weight;
    std::optional<core::TensorValue> k_bias;
    core::TensorValue v_weight;
    std::optional<core::TensorValue> v_bias;
    std::optional<core::TensorValue> qkv_weight;
    std::optional<core::TensorValue> qkv_bias;
    core::TensorValue out_weight;
    std::optional<core::TensorValue> out_bias;
};

struct RelativeAttentionWeights {
    AttentionWeights attention;
    core::TensorValue pos_weight;
    core::TensorValue pos_bias_u;
    core::TensorValue pos_bias_v;
};

struct StreamingAttentionOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

}  // namespace engine::modules
