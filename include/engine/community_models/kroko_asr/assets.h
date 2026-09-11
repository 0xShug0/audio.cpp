#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::kroko_asr {

struct KrokoASRConfig {
    std::string model_type = "zipformer2";
    std::string variant = "Kroko-Community-Streaming";
    std::string language = "auto";
    int64_t sample_rate = 16000;
    int64_t feature_dim = 80;
    int64_t chunk_size = 269;
    int64_t chunk_shift = 256;
    int64_t subsampling_factor = 4;
    int64_t vocab_size = 0;
    int64_t context_size = 2;
    int64_t blank_id = 0;
    int64_t unk_id = 2;
    std::vector<int64_t> encoder_dims;
    std::vector<int64_t> query_head_dims;
    std::vector<int64_t> value_head_dims;
    std::vector<int64_t> num_heads;
    std::vector<int64_t> num_encoder_layers;
    std::vector<int64_t> cnn_module_kernels;
    std::vector<int64_t> left_context_len;
    std::vector<int64_t> downsampling_factors;
};

struct KrokoASRAssets {
    KrokoASRConfig config;
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> weights;
    std::vector<std::string> tokens;
};

std::shared_ptr<const KrokoASRAssets> load_kroko_asr_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::kroko_asr
