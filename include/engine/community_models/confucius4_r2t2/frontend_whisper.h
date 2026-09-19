#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/types.h"

#include <memory>

namespace engine::community_models::confucius4_r2t2 {

class R2T2ASRWhisperFrontend {
public:
    explicit R2T2ASRWhisperFrontend(std::shared_ptr<const R2T2ASRAssets> assets);

    R2T2ASRAudioFeatures extract(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const R2T2ASRAssets> assets_;
    engine::audio::WhisperLogMelExtractor extractor_;
};

}  // namespace engine::community_models::confucius4_r2t2
