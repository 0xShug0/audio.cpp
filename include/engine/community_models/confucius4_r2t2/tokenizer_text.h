#pragma once

#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::confucius4_r2t2 {

class R2T2ASRTextTokenizer {
public:
    struct Impl;

    explicit R2T2ASRTextTokenizer(std::shared_ptr<const R2T2ASRAssets> assets);

    R2T2ASRPrompt build_prompt(
        const std::string & context,
        const std::string & language,
        int64_t audio_feature_tokens) const;

    R2T2ASRPrompt build_raw_audio_prompt(
        const std::string & text,
        int64_t audio_feature_tokens) const;

    /// Returns the unexpanded chat-template prompt string (audio placeholder
    /// still textual). Streaming sessions append the stable-prefix
    /// continuation before expanding the audio tokens.
    std::string build_prompt_text(const std::string & context, const std::string & language) const;

    /// Encodes plain text with the Qwen2 BPE tokenizer (added tokens such as
    /// <asr_text> resolve to their dedicated ids).
    std::vector<int32_t> encode(const std::string & text) const;

    std::string decode(const std::vector<int32_t> & token_ids) const;

private:
    std::shared_ptr<const R2T2ASRAssets> assets_;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::community_models::confucius4_r2t2
