#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/models/moss/moss_tts_nano/types.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::moss_tts_nano {

struct MossTTSNanoGlobalTransformerConfig {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t max_position_embeddings = 0;
    int64_t text_vocab_size = 0;
    float layer_norm_epsilon = 1.0e-5F;
    float rope_base = 10000.0F;
};

struct MossTTSNanoLocalTransformerConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t max_position_embeddings = 0;
    float layer_norm_epsilon = 1.0e-5F;
    float rope_base = 10000.0F;
};

struct MossTTSNanoAudioTokenizerConfig {
    std::string model_type = "moss-audio-tokenizer-nano";
    int sample_rate = 48000;
    int channels = 2;
    int64_t downsample_rate = 0;
    int64_t code_dim = 0;
    int64_t rvq_dim = 0;
    int64_t codebook_dim = 0;
    int64_t codebook_size = 1024;
    int64_t num_quantizers = 16;
};

struct MossTTSNanoConfig {
    std::string model_type;
    std::string architecture;
    int64_t n_vq = 16;
    int64_t audio_vocab_size = 1024;
    int64_t audio_pad_token_id = 1024;
    int64_t pad_token_id = 0;
    int64_t im_start_token_id = 0;
    int64_t im_end_token_id = 0;
    int64_t audio_start_token_id = 0;
    int64_t audio_end_token_id = 0;
    int64_t audio_user_slot_token_id = 0;
    int64_t audio_assistant_slot_token_id = 0;
    std::vector<int64_t> audio_codebook_sizes;
    MossTTSNanoGlobalTransformerConfig global_transformer;
    MossTTSNanoLocalTransformerConfig local_transformer;
    MossTTSNanoAudioTokenizerConfig audio_tokenizer;
};

struct MossTTSNanoAssets {
    assets::ResourceBundle resources;
    MossTTSNanoConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> audio_tokenizer_weights;
};

std::shared_ptr<const MossTTSNanoAssets> load_moss_tts_nano_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::moss_tts_nano
