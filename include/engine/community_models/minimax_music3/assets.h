#pragma once

#include "engine/community_models/minimax_music3/types.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::models::minimax_music3 {

struct MiniMaxMusic3Assets {
    std::filesystem::path model_root;
    std::filesystem::path tokenizer_json_path;
    std::filesystem::path tokenizer_config_path;
    MiniMaxMusic3Config config;
    std::shared_ptr<const assets::TensorSource> language_model_weights;
    std::shared_ptr<const assets::TensorSource> depth_decoder_weights;
    std::shared_ptr<const assets::TensorSource> condition_encoder_weights;
    std::shared_ptr<const assets::TensorSource> transformer_weights;
    std::shared_ptr<const assets::TensorSource> vocoder_weights;
};

std::shared_ptr<const MiniMaxMusic3Assets> load_minimax_music3_assets(
    const std::filesystem::path & model_path);
void validate_minimax_music3_anchors(const MiniMaxMusic3Assets & assets);

}  // namespace engine::models::minimax_music3
