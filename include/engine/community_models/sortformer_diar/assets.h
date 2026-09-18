#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::sortformer_diar {

struct SortformerV2FastConformerConfig {
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

struct SortformerV2TransformerConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t max_source_positions = 0;
    float layer_norm_eps = 1.0e-5f;
    std::string activation_function;
};

struct SortformerV2ModulesConfig {
    int64_t num_speakers = 0;
    int64_t fc_d_model = 0;
    int64_t tf_d_model = 0;
    int64_t subsampling_factor = 0;
    float dropout_rate = 0.0f;
};

struct SortformerV2StreamingConfig {
    int64_t chunk_len = 188;
    int64_t chunk_left_context = 1;
    int64_t chunk_right_context = 1;
    int64_t fifo_len = 0;
    int64_t spkcache_len = 188;
    int64_t spkcache_update_period = 188;
    int64_t spkcache_sil_frames_per_spk = 3;
    float pred_score_threshold = 0.25f;
    float scores_boost_latest = 0.05f;
    float sil_threshold = 0.2f;
    float strong_boost_rate = 0.75f;
    float weak_boost_rate = 1.5f;
    float min_pos_scores_rate = 0.5f;
    int64_t max_index = 99999;
    int64_t frame_hop_samples = 1280;
};

struct SortformerV2ModelConfig {
    std::string model_type;
    std::string variant;
    int64_t num_speakers = 0;
    float pil_weight = 0.0f;
    float ats_weight = 0.0f;
    SortformerV2FastConformerConfig fc_encoder;
    SortformerV2TransformerConfig tf_encoder;
    SortformerV2ModulesConfig modules;
    SortformerV2StreamingConfig streaming;
};

struct SortformerV2FeatureExtractorConfig {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t win_length = 0;
    int64_t hop_length = 0;
    int64_t num_mel_bins = 0;
    float preemphasis = 0.0f;
    bool return_attention_mask = true;
};

struct SortformerV2Assets {
    assets::ResourceBundle resources;
    SortformerV2ModelConfig model_config;
    SortformerV2FeatureExtractorConfig feature_config;
    // Optional trained NeMo filterbank, stored as preprocessor.fb in GGUF.
    // Empty preserves compatibility with older packages and uses the
    // deterministic audio.cpp reconstruction.
    std::vector<float> mel_filterbank;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const SortformerV2Assets> load_sortformer_v2_assets(
    const std::filesystem::path & model_root,
    const std::filesystem::path & spec_path);

}  // namespace engine::community_models::sortformer_diar
