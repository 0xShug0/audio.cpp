#pragma once

// Byte-level BPE tokenizer built from the backbone GGUF's tokenizer.ggml.*
// metadata, so the package needs no separate tokenizer files.

#include "engine/community_models/lfm2_audio/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llama_tokenizer_vendor {
struct BpeVocabulary;
}

namespace engine::community_models::lfm2_audio {

class Lfm2TextTokenizer {
public:
    explicit Lfm2TextTokenizer(const Lfm2TextVocabulary & vocabulary);

    // Special tokens such as <|im_start|> are matched whole.
    [[nodiscard]] std::vector<int32_t> encode(const std::string & text) const;
    // Drops control and unused tokens.
    [[nodiscard]] std::string decode(const std::vector<int32_t> & token_ids) const;
    [[nodiscard]] int32_t require_token_id(const std::string & token) const;
    [[nodiscard]] bool is_control_token(int32_t token_id) const;

private:
    std::shared_ptr<const llama_tokenizer_vendor::BpeVocabulary> vocab_;
    std::vector<bool> dropped_in_decode_;
};

}  // namespace engine::community_models::lfm2_audio
