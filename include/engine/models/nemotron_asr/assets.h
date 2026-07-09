#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::nemotron_asr {

struct NemotronFrontendConfig {
    int64_t sample_rate = 16000;
    int64_t feature_size = 128;
    int64_t n_fft = 512;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    float preemphasis = 0.97f;
    float log_zero_guard = 5.9604644775390625e-8f;
};

struct NemotronEncoderConfig {
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t layers = 24;
    int64_t heads = 8;
    int64_t kv_heads = 8;
    int64_t conv_kernel = 9;
    int64_t sliding_window = 57;
    int64_t subsampling_factor = 8;
    int64_t subsampling_channels = 256;
    int64_t subsampling_kernel = 3;
    int64_t subsampling_stride = 2;
    int64_t max_position_embeddings = 5000;
    int64_t default_lookahead_tokens = 3;
    std::vector<int64_t> supported_lookahead_tokens;
};

struct NemotronConfig {
    std::string model_type;
    int64_t vocab_size = 13088;
    int64_t blank_token_id = 13087;
    int64_t pad_token_id = 0;
    int64_t default_prompt_id = 101;
    int64_t decoder_hidden_size = 640;
    int64_t decoder_layers = 2;
    int64_t max_symbols_per_step = 10;
    int64_t num_prompts = 128;
    int64_t prompt_intermediate_size = 2048;
    NemotronFrontendConfig frontend;
    NemotronEncoderConfig encoder;
    std::unordered_map<std::string, int64_t> prompt_dictionary;
};

struct NemotronAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path config_path;
    std::filesystem::path processor_config_path;
    std::filesystem::path tokenizer_json_path;
    std::filesystem::path weight_path;
};

struct NemotronAssets {
    NemotronAssetPaths paths;
    NemotronConfig config;
    std::shared_ptr<const assets::TensorSource> source;
    std::shared_ptr<engine::tokenizers::HuggingFaceTokenizerJson> tokenizer;
    std::vector<uint8_t> special_token_ids;
    std::string metaspace_replacement;
    bool trim_leading_space = false;
};

NemotronAssetPaths resolve_nemotron_asr_assets(const std::filesystem::path & model_path);
std::shared_ptr<const NemotronAssets> load_nemotron_asr_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::nemotron_asr
