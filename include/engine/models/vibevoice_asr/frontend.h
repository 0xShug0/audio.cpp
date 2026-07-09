#pragma once

#include "engine/models/vibevoice_asr/assets.h"
#include "engine/framework/runtime/session.h"

#include <memory>

namespace engine::runtime {
}

namespace engine::models::vibevoice_asr {

class VibeVoiceASRFrontend {
public:
    explicit VibeVoiceASRFrontend(std::shared_ptr<const VibeVoiceAssets> assets);

    runtime::AudioBuffer normalize(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const VibeVoiceAssets> assets_;
};

}  // namespace engine::models::vibevoice_asr
