#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::sanotts {

struct SanoTtsConfig {
    int64_t vocab_size = 62;
    int64_t sample_rate = 24000;
    int64_t hop_length = 256;
    int64_t n_fft = 1024;
    int64_t mels = 100;
    int64_t dim = 0;
    int64_t blocks = 0;
    int64_t pw_hidden = 0;
    int64_t noise_channels = 4;
    int64_t dw_kernel = 7;
    int64_t embed_kernel = 7;

    int64_t duration_hidden = 0;
    int64_t duration_depth = 0;
    int64_t duration_kernel = 5;
    int64_t duration_max_tokens = 207;
    int64_t duration_max_frames = 80;

    int64_t acoustic_hidden = 0;
    int64_t acoustic_token_depth = 0;
    int64_t acoustic_depth = 0;
    int64_t acoustic_kernel = 5;

    std::string voice;
};

struct SanoTtsAssets {
    assets::ResourceBundle resources;
    SanoTtsConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const SanoTtsAssets> load_sanotts_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::sanotts
