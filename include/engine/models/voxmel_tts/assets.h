#pragma once

#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"

#include <memory>
#include <vector>

namespace engine::models::voxmel_tts {

struct ModelAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> model, vocos;
    int64_t hidden = 0, intermediate = 0, layers = 0, heads = 0, kv_heads = 0;
    int64_t vocabulary = 0, max_positions = 0;
    float norm_eps = 0, rope_theta = 0;
    int64_t mels = 0, sample_rate = 0, hop = 0;
    int64_t tier_hidden = 0, tier_heads = 0, flow_hidden = 0, flow_layers = 0;
    std::vector<int64_t> tier_layers, tier_windows;
    std::vector<char> split_axes;
    std::vector<float> mel_mean, mel_std, special, null_text;
    audio::AudioTensor mel_filterbank;
    std::vector<float> mel_window;
    bool detail_norm = false;
};

std::shared_ptr<const ModelAssets> load_assets(const std::filesystem::path & path);

} // namespace engine::models::voxmel_tts
