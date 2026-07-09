#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::hviske_asr {

struct HviskeFrontendConfig {
    int64_t sample_rate = 16000;
    int64_t features = 128;
    int64_t n_fft = 512;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    int64_t pad_to = 16;
    float dither = 1.0e-5f;
    float preemph = 0.97f;
    float log_zero_guard = 1.0f / 16777216.0f;
};

struct HviskeEncoderConfig {
    int64_t feat_in = 128;
    int64_t hidden_size = 1280;
    int64_t intermediate_size = 5120;
    int64_t layers = 48;
    int64_t heads = 8;
    int64_t conv_kernel = 9;
    int64_t subsampling_conv_channels = 256;
    int64_t subsampling_factor = 8;
    int64_t pos_emb_max_len = 5000;
};

struct HviskeDecoderConfig {
    int64_t vocab_size = 16384;
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t layers = 8;
    int64_t heads = 8;
    int64_t max_sequence_length = 1024;
    int64_t pad_token_id = 2;
    int64_t eos_token_id = 3;
    int64_t bos_token_id = 4;
    int64_t decoder_start_token_id = 13764;
    int64_t max_new_tokens = 256;
    bool output_log_probs = true;
};

struct HviskeConfig {
    std::string model_type;
    std::string variant;
    int64_t max_audio_clip_seconds = 35;
    int64_t overlap_chunk_seconds = 5;
    int64_t min_energy_window_samples = 1600;
    std::vector<std::string> supported_languages;
    HviskeFrontendConfig frontend;
    HviskeEncoderConfig encoder;
    HviskeDecoderConfig decoder;
};

struct HviskeAssets {
    std::filesystem::path model_root;
    HviskeConfig config;
    std::vector<engine::tokenizers::SentencePiecePiece> tokenizer_pieces;
    std::shared_ptr<const engine::assets::TensorSource> model_weights;
};

std::shared_ptr<const HviskeAssets> load_hviske_assets(const std::filesystem::path & model_root);

std::vector<int32_t> tokenize_hviske_prompt(
    const HviskeAssets & assets,
    const std::string & language,
    bool punctuation);

std::string decode_hviske_tokens(
    const HviskeAssets & assets,
    const std::vector<int32_t> & token_ids);

}  // namespace engine::models::hviske_asr
