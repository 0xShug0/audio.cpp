#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/modules/transformers/qwen35_decoder_runtime.h"
#include "engine/models/qwen3_asr/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::index_echo {

struct IndexEchoAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const qwen3_asr::Qwen3ASRAssets> qwen3_omni_audio;
    modules::Qwen35DecoderConfig qwen35_config;
    std::shared_ptr<const assets::TensorSource> qwen35_weights;
    std::shared_ptr<const assets::TensorSource> connector_weights;
    int64_t audio_hidden_size = 0;
    int64_t text_hidden_size = 0;
};

std::shared_ptr<const IndexEchoAssets> load_index_echo_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::index_echo
