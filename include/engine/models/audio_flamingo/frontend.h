#pragma once

#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/audio_flamingo/assets.h"
#include "engine/models/audio_flamingo/types.h"

#include <memory>

namespace engine::models::audio_flamingo {

class AudioFlamingoFrontend {
public:
    explicit AudioFlamingoFrontend(std::shared_ptr<const AudioFlamingoAssets> assets);

    AudioFlamingoAudioFeatures extract(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const AudioFlamingoAssets> assets_;
    std::shared_ptr<const engine::audio::MelSpectrogramFrontend> mel_;
};

}  // namespace engine::models::audio_flamingo
