#pragma once

// Assets & configuration for xTTS v2 (Coqui/XTTS-v2, Coqui Public Model License).
//
// Two GGUF files expected:
//   xtts-v2-gpt.gguf     — GPT-2 AR model + ConditioningEncoder + Perceiver resampler
//   xtts-v2-vocoder.gguf — HiFi-GAN vocoder + ResNet speaker encoder
//
// Architecture:
//   GPT-2: 30L, d=1024, 16 heads, head_dim=64, FFN=4096, GELU(tanh)
//   Text vocab: 6681 BPE tokens
//   Mel vocab: 1026 (1024 codes + start_mel=1024 + stop_mel=1025)
//   Text pos: 404, Mel pos: 608
//   Conditioning: AttentionBlock(6L, d=1024) -> Perceiver(2L, 32 latents, d=1024)
//   Speaker: ResNet SEBasicBlock, 64-mel input, 512-d output

#include <string>
#include <vector>
#include <filesystem>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include "engine/framework/assets/resource_bundle.h"

namespace engine::models::xtts_v2 {

struct XttSv2HyperParams {
    // GPT-2
    uint32_t n_layers = 30;
    uint32_t d_model = 1024;
    uint32_t n_heads = 16;
    uint32_t head_dim = 64;
    uint32_t ff_dim = 4096;

    // Vocab / embeddings
    uint32_t text_vocab_size = 6681;
    uint32_t mel_vocab_size = 1026;
    uint32_t text_pos_size = 404;
    uint32_t mel_pos_size = 608;

    // Special tokens
    uint32_t start_text_token = 261;
    uint32_t stop_text_token = 0;
    uint32_t start_mel_token = 1024;
    uint32_t stop_mel_token = 1025;
    uint32_t max_mel_tokens = 605;
    uint32_t code_stride = 1024;

    // Conditioning
    uint32_t cond_d_model = 1024;
    uint32_t cond_n_layers = 6;
    uint32_t cond_n_heads = 16;
    uint32_t perceiver_n_layers = 2;
    uint32_t perceiver_n_latents = 32;
};

struct XttSv2BpeVocab {
    std::vector<std::string> id_to_token;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, int32_t> token_ids;
    std::unordered_map<std::string, int> merge_ranks;
    std::vector<std::pair<std::string, int32_t>> special_tokens; // sorted by length desc

    int32_t stop_token = 0;
    int32_t unk_token = 1;
    int32_t space_token = 2;
    int32_t start_token = 261;
    std::vector<int32_t> lang_tokens;
};

struct XttSv2Assets {
    assets::ResourceBundle resources;

    std::filesystem::path gpt_gguf_path;
    std::filesystem::path vocoder_gguf_path;

    std::shared_ptr<const assets::TensorSource> gpt_source;
    std::shared_ptr<const assets::TensorSource> vocoder_source;

    XttSv2HyperParams hyper_params;
    XttSv2BpeVocab vocab;

    // Mel normalization stats from GGUF metadata or a legacy package sidecar
    std::vector<float> mel_stats;
};

// Published packages have two GGUFs; identify the GPT by its tensor schema.
std::optional<std::filesystem::path> find_xtts_v2_gpt_file(const std::filesystem::path & path);

// Load assets from a discovered model directory.
std::shared_ptr<const XttSv2Assets> load_xtts_v2_assets(const std::filesystem::path & root);

}  // namespace engine::models::xtts_v2
