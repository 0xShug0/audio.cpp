#pragma once

#include "engine/models/audio_flamingo/assets.h"
#include "engine/models/audio_flamingo/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::audio_flamingo {

class AudioFlamingoTextTokenizer {
public:
    struct Impl;

    explicit AudioFlamingoTextTokenizer(std::shared_ptr<const AudioFlamingoAssets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = true) const;
    AudioFlamingoPrompt build_prompt(const std::string & prompt, const AudioFlamingoAudioFeatures & features) const;

    int32_t audio_token_id() const noexcept;

private:
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::audio_flamingo
