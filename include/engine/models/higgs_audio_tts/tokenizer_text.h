#pragma once

#include "engine/models/higgs_audio_tts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::higgs_audio_tts {

struct HiggsPromptRequest {
    std::string text;
    std::string reference_text;
    int64_t delayed_reference_tokens = 0;
};

struct HiggsPromptEncoding {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> text_ids;
    std::vector<int32_t> reference_text_ids;
};

class HiggsTextTokenizer {
public:
    struct Impl;

    explicit HiggsTextTokenizer(std::shared_ptr<const HiggsAssets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    HiggsPromptEncoding encode_prompt(const HiggsPromptRequest & request) const;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::higgs_audio_tts
