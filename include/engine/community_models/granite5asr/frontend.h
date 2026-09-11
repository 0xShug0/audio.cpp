#pragma once

#include "engine/framework/audio/dsp.h"
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
    Granite5FrontendFeatures extract_waveform(const std::vector<float> & waveform) const;
    std::vector<float> prepare_waveform(const runtime::AudioBuffer & audio) const;

    const Granite5FrontendConfig & config() const noexcept { return assets_->config.frontend; }

private:
    std::shared_ptr<const Granite5ASRAssets> assets_;
    audio::AudioTensor mel_filterbank_;
    std::vector<float> window_;
};

}  // namespace engine::community_models::granite5asr
