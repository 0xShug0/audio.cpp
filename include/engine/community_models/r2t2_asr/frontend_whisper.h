#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/community_models/r2t2_asr/assets.h"
#include "engine/community_models/r2t2_asr/types.h"

#include <memory>

namespace engine::community_models::r2t2_asr {

class R2T2ASRWhisperFrontend {
public:
    explicit R2T2ASRWhisperFrontend(std::shared_ptr<const R2T2ASRAssets> assets);

    R2T2ASRAudioFeatures extract(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const R2T2ASRAssets> assets_;
    engine::audio::WhisperLogMelExtractor extractor_;
};

}  // namespace engine::community_models::r2t2_asr
