#pragma once

#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/granite5asr/assets.h"

#include <memory>
#include <vector>

namespace engine::community_models::granite5asr {

struct Granite5FrontendFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t feature_dim = 320;
};

class Granite5Frontend {
public:
    explicit Granite5Frontend(std::shared_ptr<const Granite5ASRAssets> assets);

    Granite5FrontendFeatures extract(const runtime::AudioBuffer & audio) const;
private:
    audio::NemoMelFrontend frontend_;
};

}  // namespace engine::community_models::granite5asr
