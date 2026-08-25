#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::audio8_tts {

struct Audio8TtsGenerationOptions {
    int64_t max_new_tokens = 1024;
    int64_t text_chunk_size = 200;
    float top_p = 0.8F;
    int top_k = 30;
    float temperature = 0.8F;
    uint32_t seed = 1234;
};

struct Audio8TtsReference {
    std::optional<runtime::AudioBuffer> audio = std::nullopt;
    std::string text;
    std::string cache_id;
};

struct Audio8TtsRequest {
    std::string text;
    std::vector<Audio8TtsReference> references;
    Audio8TtsGenerationOptions generation;
};

struct Audio8TtsCodes {
    std::vector<int32_t> codes;
    int64_t codebooks = 0;
    int64_t frames = 0;
};

struct Audio8TtsConversationTurn {
    std::string text;
    Audio8TtsCodes codes;
};

struct Audio8TtsPrompt {
    std::vector<int32_t> matrix;
    int64_t codebook_rows = 0;
    int64_t steps = 0;
    std::string text;
};

struct Audio8TtsTextConfig {
    int64_t vocab_size = 0;
    int64_t n_layer = 0;
    int64_t dim = 0;
    int64_t intermediate_size = 0;
    int64_t n_head = 0;
    int64_t n_local_heads = 0;
    int64_t head_dim = 0;
    int64_t max_seq_len = 0;
    float rope_base = 1000000.0F;
    float norm_eps = 1.0e-6F;
    bool tie_word_embeddings = true;
    bool attention_qk_norm = true;
};

struct Audio8TtsFastConfig {
    int64_t vocab_size = 0;
    int64_t num_codebooks = 0;
    int64_t n_layer = 0;
    int64_t dim = 0;
    int64_t intermediate_size = 0;
    int64_t n_head = 0;
    int64_t n_local_heads = 0;
    int64_t head_dim = 0;
    int64_t max_seq_len = 0;
    float rope_base = 1000000.0F;
    float norm_eps = 1.0e-6F;
    bool tie_word_embeddings = false;
    bool attention_qk_norm = false;
};

struct Audio8TtsCodecConfig {
    int sample_rate = 44100;
    int64_t semantic_codebook_size = 4096;
    int64_t residual_codebook_size = 1024;
    int64_t quantizer_codebooks = 9;
    int64_t total_codebooks = 10;
    int64_t codebook_dim = 8;
    int64_t latent_dim = 1024;
    int64_t frame_length = 2048;
};

struct Audio8TtsConfig {
    std::string model_type;
    std::string torch_dtype;
    int64_t semantic_start_token_id = 0;
    int64_t semantic_end_token_id = 0;
    int64_t im_end_token_id = 0;
    bool norm_fastlayer_input = false;
    Audio8TtsTextConfig text;
    Audio8TtsFastConfig fast;
    Audio8TtsCodecConfig codec;
};

}  // namespace engine::models::audio8_tts
