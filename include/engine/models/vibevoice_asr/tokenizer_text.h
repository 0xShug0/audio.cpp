#pragma once

#include "engine/models/vibevoice_asr/assets.h"
#include "engine/models/vibevoice_asr/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::vibevoice_asr {

class VibeVoiceASRTextTokenizer {
public:
    struct Impl;

    explicit VibeVoiceASRTextTokenizer(std::shared_ptr<const VibeVoiceASRAssets> assets);

    VibeVoiceASRPrompt build_prompt(const VibeVoiceASRRequest & request, int64_t speech_tokens) const;
    std::vector<int32_t> build_streaming_prompt(const std::string & context) const;
    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const;

    int32_t eos_id() const noexcept;
    int32_t pad_id() const noexcept;
    int32_t speech_start_id() const noexcept;
    int32_t speech_end_id() const noexcept;
    int32_t speech_pad_id() const noexcept;
    int32_t text_chunk_end_id() const noexcept;

private:
    std::shared_ptr<const VibeVoiceASRAssets> assets_;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::vibevoice_asr
