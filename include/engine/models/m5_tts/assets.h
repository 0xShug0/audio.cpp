#pragma once

#include "engine/framework/model_spec/package.h"
#include "engine/framework/codecs/qwen3_audio_tokenizer_runtime.h"

#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::m5_tts {

struct DecoderConfig {
    int64_t layers = 0;
    int64_t hidden = 0;
    int64_t intermediate = 0;
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    float norm_eps = 1.0e-5F;
    float rope_theta = 500000.0F;
};

struct ModelAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> model;
    std::shared_ptr<const assets::TensorSource> codec;
    DecoderConfig backbone;
    DecoderConfig decoder;
    codecs::Qwen3AudioTokenizerDecoderConfig codec_config;
    int64_t text_vocabulary = 0;
    int64_t audio_vocabulary = 0;
    int64_t codebooks = 0;
    int64_t context_length = 0;
    int32_t bos_token_id = 0;
    int32_t eos_token_id = 0;
    std::string variant;
};

std::shared_ptr<const ModelAssets> load_assets(const std::filesystem::path & path);

}  // namespace engine::models::m5_tts
