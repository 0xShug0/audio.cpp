#pragma once

#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/models/confucius4_tts/assets.h"
#include "engine/models/confucius4_tts/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::confucius4_tts {

class ConfuciusTextTokenizer {
public:
    explicit ConfuciusTextTokenizer(std::shared_ptr<const ConfuciusAssets> assets);

    std::string normalize(const std::string & text, const std::string & language) const;
    std::string format_prompt(const std::string & text, const std::string & language) const;
    std::vector<int32_t> encode(const std::string & text) const;
    std::vector<int32_t> encode_without_special_tokens(const std::string & text) const;
    std::vector<int32_t> encode_prompt(const std::string & text, const std::string & language) const;
    std::vector<ConfuciusTextSegment> segment_request(const ConfuciusRequest & request) const;

private:
    std::shared_ptr<const ConfuciusAssets> assets_;
    std::vector<engine::tokenizers::SentencePiecePiece> pieces_;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
};

}  // namespace engine::models::confucius4_tts
