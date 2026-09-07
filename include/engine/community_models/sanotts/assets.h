#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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

enum class SanoTtsGraph {
    Nano,        // mel-100 -> ConvNeXt-1D -> iSTFT, noise-fed (heart, heart-nano)
    Piperlite,   // 192-ch latent -> 3-stage ConvTranspose1d, deterministic (amy, ...)
};

struct SanoTtsPiperConfig {
    std::string voice;
    std::string language;       // short code the session validates against: en, vi, id
    std::string espeak_voice;
    int64_t sample_rate = 22050;
    double duration_length_scale = 1.0;

    int64_t duration_vocab = 0;
    int64_t duration_hidden = 0;
    int64_t duration_depth = 0;
    int64_t duration_kernel = 5;
    int64_t duration_max_tokens = 0;
    int64_t duration_max_frames = 0;

    int64_t acoustic_vocab = 0;
    int64_t acoustic_hidden = 0;
    int64_t acoustic_depth = 0;
    int64_t acoustic_token_depth = 0;
    int64_t acoustic_kernel = 5;
    int64_t acoustic_out_channels = 0;

    std::array<int64_t, 4> channels = {0, 0, 0, 0};
    std::array<std::vector<int64_t>, 3> stage_branches;
    int64_t post_filter_channels = 0;
    int64_t post_filter_layers = 0;
    int64_t post_filter_kernel = 9;
    double post_filter_scale = 0.0;

    /** Piper phoneme_id_map: one UTF-8 codepoint -> id. */
    std::unordered_map<std::string, int32_t> phoneme_id_map;
};

struct SanoTtsAssets {
    assets::ResourceBundle resources;
    SanoTtsGraph graph = SanoTtsGraph::Nano;
    SanoTtsConfig config;      // valid when graph == Nano
    SanoTtsPiperConfig piper;  // valid when graph == Piperlite
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const SanoTtsAssets> load_sanotts_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::sanotts
