#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/index_tts2_5/types.h"

#include <filesystem>
#include <memory>

namespace engine::models::index_tts2_5 {

struct IndexTTS25Assets {
    assets::ResourceBundle resources;
    IndexTTS25Config config;
    std::shared_ptr<const assets::TensorSource> gpt_weights;
    std::shared_ptr<const assets::TensorSource> s2mel_weights;
    std::shared_ptr<const assets::TensorSource> speaker_matrix;
    std::shared_ptr<const assets::TensorSource> emotion_matrix;
    std::shared_ptr<const assets::TensorSource> wav2vec2bert_stats;
    std::shared_ptr<const assets::TensorSource> wav2vec2bert_weights;
    std::shared_ptr<const assets::TensorSource> semantic_codec_weights;
    std::shared_ptr<const assets::TensorSource> campplus_weights;
    std::shared_ptr<const assets::TensorSource> bigvgan_weights;
    std::shared_ptr<const assets::TensorSource> qwen_emotion_weights;
};

std::shared_ptr<const IndexTTS25Assets> load_index_tts2_5_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::index_tts2_5
