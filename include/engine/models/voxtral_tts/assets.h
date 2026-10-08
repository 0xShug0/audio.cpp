#pragma once

#include "engine/framework/model_spec/package.h"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::voxtral_tts {

constexpr size_t kContextBytes = 4 * 1024 * 1024;
constexpr int64_t kAudioTokenOffset = 2;
constexpr int64_t kAudioEndToken = 1;

struct TransformerConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t layers = 0;
    float norm_eps = 1e-5F;
    float rope_theta = 10000.0F;
};

struct VoxtralTtsConfig {
    TransformerConfig ar;
    TransformerConfig acoustic;
    TransformerConfig codec;
    int64_t vocab_size = 0;
    int64_t semantic_vocab_size = 0;
    int64_t semantic_dim = 0;
    int64_t acoustic_dim = 0;
    int64_t acoustic_levels = 0;
    int64_t sample_rate = 0;
    int64_t patch_size = 0;
    int64_t patch_kernel_size = 0;
    float codec_qk_norm_eps = 1e-6F;
    std::vector<int> codec_layers;
    std::vector<int> codec_kernels;
    std::vector<int> codec_strides;
    std::vector<int> codec_windows;
};

struct VoxtralTtsAssets {
    assets::ResourceBundle resources;
    VoxtralTtsConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
    std::map<std::string, std::vector<float>> voices;
};

std::shared_ptr<const VoxtralTtsAssets> load_assets(const std::filesystem::path & path);

}  // namespace engine::models::voxtral_tts
