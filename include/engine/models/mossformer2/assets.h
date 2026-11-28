#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::mossformer2 {

struct MossFormer2Implementation;

struct MossFormer2VocoderConfig {
    int initial_channels = 1024;
    std::vector<int64_t> upsample_rates, upsample_kernels, resblock_kernels;
    std::vector<std::vector<int64_t>> resblock_dilations;
};

struct MossFormer2Config {
    int sample_rate = 16000;
    int speakers = 2;
    int kernel = 16;
    int channels = 512;
    int hidden = 512;
    int layers = 24;
    int output_features = 0;
    int fft_length = 0;
    int hop_length = 0;
    int mel_bins = 0;
    float mel_low = 0;
    float mel_high = 0;
    MossFormer2VocoderConfig vocoder;
};

struct MossFormer2Assets {
    const MossFormer2Implementation * impl = nullptr;
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensors;
    MossFormer2Config config;
};

std::shared_ptr<const MossFormer2Assets> load_mossformer2_assets(const std::filesystem::path & path);

}  // namespace engine::models::mossformer2
