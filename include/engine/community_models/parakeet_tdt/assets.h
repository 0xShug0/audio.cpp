#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct ParakeetFrontendConfig {
    int64_t sample_rate = 16000;
    int64_t feature_size = 128;
    int64_t n_fft = 512;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    float preemphasis = 0.97f;
    float log_zero_guard = 5.9604644775390625e-8f;
};

struct ParakeetEncoderConfig {
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t layers = 24;
    int64_t heads = 8;
    int64_t conv_kernel = 9;
    int64_t subsampling_factor = 8;
    int64_t subsampling_channels = 256;
    int64_t subsampling_kernel = 3;
    int64_t subsampling_stride = 2;
    int64_t max_position_embeddings = 5000;
};

struct ParakeetConfig {
    std::string model_type;
    int64_t vocab_size = 8193;
    int64_t blank_token_id = 8192;
    int64_t pad_token_id = 2;
    int64_t decoder_hidden_size = 640;
    int64_t decoder_layers = 2;
    int64_t max_symbols_per_step = 10;
    std::vector<int32_t> durations = {0, 1, 2, 3, 4};
    ParakeetFrontendConfig frontend;
    ParakeetEncoderConfig encoder;
};

struct ParakeetTDTAssets {
    assets::ResourceBundle resources;
    ParakeetConfig config;
    std::shared_ptr<const assets::TensorSource> source;
    std::shared_ptr<engine::tokenizers::HuggingFaceTokenizerJson> tokenizer;
    std::vector<uint8_t> special_token_ids;
};

std::shared_ptr<const ParakeetTDTAssets> load_parakeet_assets(const std::filesystem::path & model_path);

}  // namespace engine::community_models::parakeet_tdt
