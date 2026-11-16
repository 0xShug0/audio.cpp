#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::tiny_voca {

// The model settings of config.json (the "model" object of the release).
struct TinyVocaConfig {
    int64_t sample_rate = 24000;
    int64_t hop = 256;
    float f0_min = 50.0F;
    float f0_max = 600.0F;
    std::vector<std::string> voices;
    std::vector<std::string> symbols;       // the phoneme inventory, by id
    int64_t vocab_size = 0;
    int64_t channels = 0;
    int64_t heads = 0;
    int64_t feedforward = 0;
    int64_t kernel = 0;
    int64_t encoder_layers = 0;
    int64_t prosody_layers = 0;
    int64_t acoustic_layers = 0;
    int64_t predictor_channels = 0;
    int64_t synth_fft = 0;
    int64_t synth_hop = 0;
    int64_t noise_fft = 0;
    int64_t noise_hop = 0;
    int64_t cep_causal = 0;
    int64_t cep_anti = 0;
    int64_t noise_cep = 0;
    float harmonic_limit = 0.47F;
    int64_t max_tokens = 512;
    int64_t max_frames = 3000;
    float pitch_reference_hz = 200.0F;
    int64_t phonemizer_version = 0;
};

struct TinyVocaAssets {
    assets::ResourceBundle resources;
    TinyVocaConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
    std::shared_ptr<const assets::TensorSource> g2p_weights;
};

std::shared_ptr<const TinyVocaAssets> load_tiny_voca_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::tiny_voca
