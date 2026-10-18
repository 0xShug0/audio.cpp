#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::personaplex {

struct PersonaPlexMimiConfig {
    int sample_rate = 24000;
    float frame_rate = 12.5F;
    int64_t channels = 1;
    int64_t hidden_size = 512;
    int64_t num_heads = 8;
    int64_t intermediate_size = 2048;
    int64_t transformer_layers = 8;
    int64_t context = 250;
    int64_t latent_size = 256;
    int64_t codebooks = 8;
    int64_t total_codebooks = 32;
    int64_t codebook_size = 2048;
    int64_t encoder_upsample_stride = 16;
};

struct PersonaPlexLMConfig {
    int64_t hidden_size = 4096;
    int64_t num_layers = 32;
    int64_t num_attention_heads = 32;
    int64_t num_key_value_heads = 32;
    int64_t head_dim = 128;
    int64_t intermediate_size = 11264;
    int64_t text_vocab_size = 32000;
    int64_t text_padding_token_id = 3;
    int64_t lm_codebooks = 16;
    int64_t voice_codebooks = 8;
    int64_t depformer_steps = 16;
    int64_t audio_codebook_size = 2048;
    int64_t context = 3000;
    float rms_norm_eps = 1.0e-8F;
    float rope_theta = 10000.0F;
};

struct PersonaPlexDepformerConfig {
    int64_t hidden_size = 1024;
    int64_t num_layers = 6;
    int64_t num_attention_heads = 16;
    int64_t head_dim = 64;
    int64_t intermediate_size = 2816;
    int64_t context = 8;
};

struct PersonaPlexConfig {
    std::string model_type = "personaplex";
    std::string version = "7b-v1";
    PersonaPlexMimiConfig mimi;
    PersonaPlexLMConfig lm;
    PersonaPlexDepformerConfig depformer;
};

struct PersonaPlexVoicePrompt {
    std::string id;
    std::shared_ptr<const assets::TensorSource> source;
};

struct PersonaPlexAssets {
    assets::ResourceBundle resources;
    PersonaPlexConfig config;
    std::vector<engine::tokenizers::SentencePiecePiece> tokenizer_pieces;
    std::shared_ptr<const assets::TensorSource> lm_weights;
    std::shared_ptr<const assets::TensorSource> mimi_weights;
    std::unordered_map<std::string, std::filesystem::path> voice_prompt_paths;
};

std::shared_ptr<const PersonaPlexAssets> load_personaplex_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::personaplex
