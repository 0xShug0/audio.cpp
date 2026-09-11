#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/models/higgs_audio_stt/assets.h"
#include "engine/models/higgs_audio_stt/types.h"

#include <memory>

namespace engine::models::higgs_audio_stt {

class HiggsAudioSTTWhisperFrontend {
public:
    explicit HiggsAudioSTTWhisperFrontend(std::shared_ptr<const HiggsAudioSTTAssets> assets);

    HiggsAudioSTTAudioFeatures extract(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const HiggsAudioSTTAssets> assets_;
    engine::audio::WhisperLogMelExtractor extractor_;
};

}  // namespace engine::models::higgs_audio_stt
