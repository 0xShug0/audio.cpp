#pragma once

#include "engine/models/ming3_audio/talker.h"
#include "engine/framework/tokenizers/llama_bpe.h"

namespace engine::models::ming3_audio {

class TalkerFrontend {
public:
    explicit TalkerFrontend(const MingAudioAssets & assets);

    TalkerPrompt build(const std::string & text, const std::string & voice, const std::string & language,
                       bool reference, const std::string & reference_text = {}, int64_t reference_patches = 0) const;

private:
    tokenizers::LlamaBpeTokenizer tokenizer_;
};

}  // namespace engine::models::ming3_audio
