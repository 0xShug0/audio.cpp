#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

namespace engine::models::nemotron_3_diar {

struct EncoderConfig {
    int64_t feature_size = 128;
    int64_t hidden_size = 512;
    int64_t intermediate_size = 2048;
    int64_t heads = 8;
    int64_t layers = 31;
    int64_t subsampling_factor = 8;
    float rope_theta = 10000.0F;
    float layer_norm_eps = 1.0e-5F;
};

struct HeadConfig {
    int64_t hidden_size = 192;
    int64_t upsample_factor = 8;
    int64_t upsample_kernel_size = 3;
};

struct StreamingConfig {
    int64_t spkcache_len = 264;
    int64_t fifo_len = 0;
    int64_t chunk_len = 264;
    int64_t chunk_left_context = 0;
    int64_t chunk_right_context = 0;
    int64_t spkcache_update_period = 188;
    int64_t spkcache_sil_frames_per_spk = 1;
    float pred_score_threshold = 0.25F;
    float scores_boost_latest = 0.05F;
    float sil_threshold = 0.2F;
    float strong_boost_rate = 0.75F;
    float weak_boost_rate = 1.5F;
    float min_pos_scores_rate = 0.5F;
    int64_t max_index = 99999;
};

struct FeatureConfig {
    int64_t sample_rate = 16000;
    int64_t n_fft = 512;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    int64_t num_mel_bins = 128;
    float preemphasis = 0.97F;
    float dither = 1.0e-5F;
};

struct ModelConfig {
    int64_t num_speakers = 8;
    bool high_resolution = true;
    int64_t output_subsampling_factor = 1;
    EncoderConfig encoder;
    HeadConfig head;
    StreamingConfig streaming;
};

struct EncoderLayerWeights {
    modules::NormWeights norm1;
    modules::AttentionWeights attention;
    modules::NormWeights norm2;
    modules::LinearWeights ffn_in;
    modules::LinearWeights ffn_out;
};

struct ModelWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights pre_encode;
    modules::NormWeights embed_norm;
    std::vector<EncoderLayerWeights> layers;
    modules::NormWeights final_norm;
    modules::LinearWeights encoder_projection;
    modules::Conv1dWeights subpixel_upsample;
    modules::LinearWeights head_hidden;
    modules::LinearWeights speaker_head;
};

struct Assets {
    assets::ResourceBundle resources;
    ModelConfig model_config;
    FeatureConfig feature_config;
    std::vector<float> mel_filterbank;
    std::vector<float> silence_embedding;
    std::shared_ptr<const audio::NemoMelFrontend> frontend;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const Assets> load_assets(
    const std::filesystem::path & model_root,
    const std::filesystem::path & spec_path);

std::shared_ptr<ModelWeights> load_weights(
    const Assets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type,
    size_t context_bytes);

}  // namespace engine::models::nemotron_3_diar
