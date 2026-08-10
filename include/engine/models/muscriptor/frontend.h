#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/muscriptor/assets.h"
#include "engine/models/muscriptor/types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::muscriptor {

class MuScriptorFrontend {
public:
    MuScriptorFrontend(std::shared_ptr<const MuScriptorAssets> assets, size_t threads);

    std::vector<MuScriptorAudioChunk> extract_chunks(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const MuScriptorAssets> assets_;
    engine::audio::SparseMelFilterbank filterbank_;
    size_t threads_ = 0;
};

}  // namespace engine::models::muscriptor
