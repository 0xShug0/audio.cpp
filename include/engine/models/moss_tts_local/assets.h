#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::moss_tts_local {

struct MossBackboneConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
};

struct MossLocalTransformerConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t max_positions = 0;
    float layer_norm_eps = 1.0e-6F;
    float rope_base = 1000000.0F;
};

struct MossTTSLocalConfig {
    MossBackboneConfig backbone;
    MossLocalTransformerConfig local;
    int64_t num_codebooks = 0;
    int64_t audio_vocab_size = 0;
    std::vector<int64_t> audio_codebook_sizes;
    int64_t audio_pad_token_id = 0;
    int64_t pad_token_id = 0;
    int64_t im_start_token_id = 0;
    int64_t im_end_token_id = 0;
    int64_t audio_start_token_id = 0;
    int64_t audio_end_token_id = 0;
    int64_t audio_user_slot_token_id = 0;
    int64_t audio_assistant_slot_token_id = 0;
    int64_t sampling_rate = 0;
    std::string local_text_head_mode;
    std::string audio_tokenizer_name_or_path;
};

struct MossTTSLocalAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path config_path;
    std::filesystem::path model_weights_path;
    std::optional<std::filesystem::path> tokenizer_json_path;
    std::optional<std::filesystem::path> tokenizer_config_path;
    std::optional<std::filesystem::path> tokenizer_vocab_path;
    std::optional<std::filesystem::path> tokenizer_merges_path;
    std::optional<std::filesystem::path> audio_tokenizer_root;
};

struct MossTTSLocalAssets {
    MossTTSLocalAssetPaths paths;
    MossTTSLocalConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

MossTTSLocalAssetPaths resolve_moss_tts_local_assets(const std::filesystem::path & model_path);
MossTTSLocalConfig load_moss_tts_local_config(const std::filesystem::path & model_path);
std::shared_ptr<const MossTTSLocalAssets> load_moss_tts_local_assets(const std::filesystem::path & model_path);

// Locates the MOSS-Audio-Tokenizer-v2 codec weights: prefers a local audio_tokenizer/
// directory next to the model, otherwise resolves the Hugging Face hub cache snapshot for
// the repo id recorded in config (audio_tokenizer_name_or_path).
std::filesystem::path resolve_moss_codec_dir(const MossTTSLocalAssets & assets);

}  // namespace engine::models::moss_tts_local
