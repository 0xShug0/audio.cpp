#pragma once

#include "engine/models/audio_flamingo/assets.h"
#include "engine/models/audio_flamingo/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::audio_flamingo {

class AudioFlamingoTextTokenizer {
public:
    struct Impl;

    explicit AudioFlamingoTextTokenizer(std::shared_ptr<const AudioFlamingoAssets> assets);

    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = true) const;
    AudioFlamingoPrompt build_prompt(const std::string & prompt, const AudioFlamingoAudioFeatures & features) const;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::audio_flamingo
