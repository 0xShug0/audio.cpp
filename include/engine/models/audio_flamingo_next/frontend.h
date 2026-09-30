#pragma once

#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/models/audio_flamingo_next/assets.h"
#include "engine/models/audio_flamingo_next/types.h"

#include <memory>

namespace engine::models::audio_flamingo_next {

class AFNextFrontend {
public:
    explicit AFNextFrontend(std::shared_ptr<const AFNextAssets> assets);

    AFNextAudioFeatures extract(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const AFNextAssets> assets_;
    std::shared_ptr<const engine::audio::MelSpectrogramFrontend> mel_;
};

}  // namespace engine::models::audio_flamingo_next
