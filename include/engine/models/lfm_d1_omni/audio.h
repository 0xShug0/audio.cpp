#pragma once

#include "engine/models/lfm_d1_omni/assets.h"
#include "engine/framework/runtime/session.h"

namespace engine::models::lfm_d1_omni {

class D1AudioFrontend {
public:
    explicit D1AudioFrontend(LfmD1OmniAudioSpec config);
    audio::MelSpectrogramFeatures extract(const runtime::AudioBuffer & input, int threads);

private:
    LfmD1OmniAudioSpec config_;
    audio::MelSpectrogramFrontend frontend_;
};

} // namespace engine::models::lfm_d1_omni
