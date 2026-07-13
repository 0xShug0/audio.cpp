#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::moss {

struct TokenRows {
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> audio_codes;
};

class TokenRowBuilder {
public:
    TokenRowBuilder(int64_t num_codebooks, int32_t audio_pad_token_id);

    void push_text_token(int32_t token_id);
    void push_text_tokens(const std::vector<int32_t> & token_ids);
    void push_audio_row(int32_t text_slot_token_id, const std::vector<std::vector<int32_t>> & codes, int64_t frame);
    TokenRows finish();

private:
    int64_t num_codebooks_ = 0;
    int32_t audio_pad_token_id_ = 0;
    TokenRows rows_;
};

struct AudioCodebookSpec {
    int64_t hidden_size = 0;
    int64_t num_codebooks = 0;
    int64_t audio_vocab_size = 0;
    int64_t audio_pad_token_id = 0;
    std::vector<int64_t> audio_codebook_sizes;
    std::string tensor_prefix = "audio_embeddings";
};

class AudioCodebookEmbeddings {
public:
    AudioCodebookEmbeddings(const assets::TensorSource & source, AudioCodebookSpec spec);

    int64_t hidden_size() const noexcept { return hidden_size_; }
    int64_t num_codebooks() const noexcept { return num_codebooks_; }
    int32_t audio_pad_token_id() const noexcept { return audio_pad_token_id_; }
    int64_t codebook_size(int64_t codebook) const;
    const float * embedding(int64_t codebook, int32_t code) const;
    void add_bias(const int32_t * codes, float * bias) const;
    std::vector<float> bias_for(const int32_t * codes) const;

private:
    int64_t hidden_size_ = 0;
    int64_t num_codebooks_ = 0;
    int32_t audio_pad_token_id_ = 0;
    std::vector<std::vector<float>> embeddings_;
};

}  // namespace engine::models::moss
