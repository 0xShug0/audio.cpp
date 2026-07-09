#pragma once

#include "engine/models/higgs_audio_stt/assets.h"
#include "engine/models/higgs_audio_stt/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::higgs_audio_stt {

class HiggsAudioSTTTextTokenizer {
public:
    struct Impl;

    explicit HiggsAudioSTTTextTokenizer(std::shared_ptr<const HiggsAudioSTTAssets> assets);

    HiggsAudioSTTPrompt build_prompt(
        const std::string & context,
        const std::string & language,
        bool enable_thinking,
        int64_t audio_feature_tokens) const;

    HiggsAudioSTTPrompt build_raw_audio_prompt(
        const std::string & text,
        int64_t audio_feature_tokens) const;

    std::string decode(const std::vector<int32_t> & token_ids) const;

private:
    std::shared_ptr<const HiggsAudioSTTAssets> assets_;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::higgs_audio_stt
