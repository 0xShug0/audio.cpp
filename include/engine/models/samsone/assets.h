#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::samsone {

struct SamsoneConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t num_hidden_layers = 0;
    int64_t vocab_size = 0;
    int64_t max_position_embeddings = 0;
    float rms_norm_eps = 1.0e-5F;
    float rope_theta = 100000.0F;
    int32_t eos_token_id = 0;
};

struct SamsoneAssets {
    SamsoneConfig config;
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> weights;
};

std::shared_ptr<const SamsoneAssets> load_samsone_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::samsone
