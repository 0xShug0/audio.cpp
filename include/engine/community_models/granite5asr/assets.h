#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::granite5asr {

struct Granite5FrontendConfig {
    int64_t sample_rate = 16000;
    int64_t n_fft = 512;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    int64_t n_mels = 80;
    int64_t stack_factor = 2;
    bool deltas = true;
    int64_t delta_win_length = 3;
    float logmel_floor_db = 8.0f;
};

struct Granite5EncoderConfig {
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t num_layers = 16;
    int64_t num_attention_heads = 8;
    int64_t num_key_value_heads = 8;
    int64_t head_dim = 128;
    int64_t context_size = 128;
    int64_t conv_kernel_size = 7;
    int64_t conv_expansion_factor = 2;
    int64_t max_position_embeddings = 512;
    int64_t num_mel_bins = 80;
    int64_t input_features = 320;
    int64_t vocab_size = 16384;
    std::vector<int64_t> subsample_layers = {0, 1};
};

struct Granite5ASRConfig {
    std::string model_type = "granite_speech5_ctc";
    int64_t vocab_size = 16384;
    int64_t pad_token_id = 0;
    int64_t blank_token_id = 0;
    Granite5FrontendConfig frontend;
    Granite5EncoderConfig encoder;
};

struct Granite5ASRAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    Granite5ASRConfig config;
    std::shared_ptr<tokenizers::HuggingFaceTokenizerJson> tokenizer;
    std::vector<uint8_t> special_token_ids;
};

Granite5ASRConfig parse_granite5asr_config(const std::string & json_text);

std::shared_ptr<const Granite5ASRAssets> load_granite5asr_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::community_models::granite5asr
