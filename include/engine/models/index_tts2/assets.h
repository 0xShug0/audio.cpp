#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/index_tts2/types.h"

#include <filesystem>
#include <memory>

namespace engine::models::index_tts2 {

struct IndexTTS2AssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path config_yaml_path;
    std::filesystem::path bpe_model_path;
    std::filesystem::path gpt_weights_path;
    std::filesystem::path s2mel_weights_path;
    std::filesystem::path speaker_matrix_path;
    std::filesystem::path emotion_matrix_path;
    std::filesystem::path wav2vec2bert_stats_path;
    std::filesystem::path wav2vec2bert_root;
    std::filesystem::path wav2vec2bert_config_path;
    std::filesystem::path wav2vec2bert_preprocessor_config_path;
    std::filesystem::path wav2vec2bert_weights_path;
    std::filesystem::path semantic_codec_weights_path;
    std::filesystem::path campplus_weights_path;
    std::filesystem::path bigvgan_root;
    std::filesystem::path bigvgan_config_path;
    std::filesystem::path bigvgan_weights_path;
    std::filesystem::path qwen_emotion_root;
    std::filesystem::path qwen_emotion_config_path;
    std::filesystem::path qwen_emotion_generation_config_path;
    std::filesystem::path qwen_emotion_tokenizer_json_path;
    std::filesystem::path qwen_emotion_tokenizer_config_path;
    std::filesystem::path qwen_emotion_vocab_path;
    std::filesystem::path qwen_emotion_merges_path;
    std::filesystem::path qwen_emotion_weights_path;
};

struct IndexTTS2Assets {
    IndexTTS2AssetPaths paths;
    IndexTTS2Config config;
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

IndexTTS2AssetPaths resolve_index_tts2_assets(const std::filesystem::path & model_path);
std::shared_ptr<const IndexTTS2Assets> load_index_tts2_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::index_tts2
