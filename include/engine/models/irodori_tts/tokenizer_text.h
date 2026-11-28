#pragma once

#include "engine/models/irodori_tts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::irodori_tts {

struct IrodoriTokenizedText {
    std::vector<int32_t> token_ids;
    std::vector<uint8_t> mask;
};

class IrodoriTextTokenizer {
public:
    explicit IrodoriTextTokenizer(std::shared_ptr<const IrodoriTTSAssets> assets);

    // The text to speak. With the v4 Small tokenizer, rare long pieces are left
    // out, as in Python Irodori-TTS (see tokenizer_text.cpp).
    IrodoriTokenizedText encode_padded(const std::string & text, int64_t max_length) const;
    std::vector<int32_t> encode(const std::string & text) const;
    // The caption, with every piece of the tokenizer: Python does not split captions.
    IrodoriTokenizedText encode_caption_padded(const std::string & caption, int64_t max_length) const;
    std::vector<int32_t> encode_caption(const std::string & caption) const;
    int32_t bos_id() const noexcept;
    int32_t pad_id() const noexcept;
    int32_t unk_id() const noexcept;
    int64_t vocab_size() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::irodori_tts
