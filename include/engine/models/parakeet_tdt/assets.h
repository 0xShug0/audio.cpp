#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::tokenizers {
class HuggingFaceTokenizerJson;
}

namespace engine::models::parakeet_tdt {

struct ParakeetFeatureExtractorConfig {
    int64_t feature_size = 0;
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t win_length = 0;
    int64_t hop_length = 0;
    float preemphasis = 0.0f;
    bool return_attention_mask = true;
};

struct ParakeetEncoderConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_mel_bins = 0;
    int64_t max_position_embeddings = 0;
    int64_t conv_kernel_size = 0;
    int64_t subsampling_factor = 0;
    int64_t subsampling_conv_channels = 0;
    int64_t subsampling_conv_kernel_size = 0;
    int64_t subsampling_conv_stride = 0;
    bool attention_bias = false;
    bool convolution_bias = false;
    bool scale_input = false;
    std::string hidden_act;
};

struct ParakeetModelConfig {
    std::string model_type;
    std::string variant;
    int64_t blank_token_id = 0;
    int64_t pad_token_id = 0;
    int64_t vocab_size = 0;
    int64_t decoder_hidden_size = 0;
    int64_t num_decoder_layers = 0;
    int64_t max_symbols_per_step = 0;
    std::vector<int32_t> durations;
    ParakeetEncoderConfig encoder;
};

struct ParakeetAssets {
    ParakeetModelConfig model_config;
    ParakeetFeatureExtractorConfig feature_config;
    std::shared_ptr<engine::tokenizers::HuggingFaceTokenizerJson> tokenizer;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const ParakeetAssets> load_parakeet_assets(const std::filesystem::path & model_root);
ParakeetModelConfig load_parakeet_model_config(const std::filesystem::path & path);
ParakeetFeatureExtractorConfig load_parakeet_feature_config(const std::filesystem::path & path);
std::shared_ptr<engine::tokenizers::HuggingFaceTokenizerJson> load_parakeet_tokenizer(const std::filesystem::path & path);

}  // namespace engine::models::parakeet_tdt
